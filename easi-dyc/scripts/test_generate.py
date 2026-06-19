# common utils
import os
import argparse
import time

# pytorch3d
from pytorch3d.renderer import TexturesUV

# torch
import torch

from torchvision import transforms
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

# diffusers (real inpainting)
from diffusers import ControlNetModel, DDIMScheduler, StableDiffusionControlNetInpaintPipeline

# numpy
import numpy as np

# image
import cv2
from PIL import Image

# customized
import sys
sys.path.append(".")

from lib.mesh_helper import (
    init_mesh,
    adjust_uv_map,
)
from lib.render_helper import render
from lib.io_helper import (
    save_backproject_obj,
    save_args,
    save_viewpoints
)
from lib.vis_helper import (
    visualize_principle_viewpoints,
    visualize_refinement_viewpoints
)
from lib.diffusion_helper import (
    get_inpainting,
    apply_inpainting_postprocess
)
from lib.projection_helper import (
    backproject_from_image,
    render_one_view_and_build_masks,
    select_viewpoint,
    build_similarity_texture_cache_for_all_views
)
from lib.camera_helper import init_viewpoints
from lib.dataloader_utils import ImageUtils
from ip_adapter.resampler import Resampler
from ip_adapter.ip_adapter import setup_ipadapter_attention_processors
from ip_adapter.attention_processor import IPAttnProcessor, IPAttnProcessor2_0


if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(DEVICE)
else:
    print("no gpu avaiable")
    raise SystemExit(1)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def init_args():
    """
    目标：用你提供的 view0 彩色结果作为“初始纹理种子”，然后对其余视角走“基于当前纹理渲染底图的 inpainting 式生成”，
    每一视角生成后立即 back-project 更新纹理，再进入下一视角，从而提升多视角一致性。
    """
    parser = argparse.ArgumentParser()
    # === 让脚本可直接运行：将你给的命令行配置写成默认值（仍可通过命令行覆盖） ===
    parser.add_argument("--input_dir", type=str, default="data/meshes/cat")
    parser.add_argument("--output_dir", type=str, default="outputs_new_02")
    parser.add_argument("--obj_file", type=str, default="cat.obj")
    parser.add_argument("--prompt", type=str, default="American Shorthair,yellow-green eyes,A cat with black and silver stripes")
    parser.add_argument("--a_prompt", type=str, default="best quality, high quality, extremely detailed, good geometry")
    parser.add_argument("--n_prompt", type=str, default="deformed, extra digit, fewer digits, cropped, worst quality, low quality, smoke")

    # 关键：inpainting 强度（越小越依赖底图 = 当前纹理渲染结果）
    parser.add_argument("--inpaint_strength", type=float, default=1,
                        help="Inpainting-like strength for view>0 generation (lower => follow current rendered init_image more).")
    parser.add_argument("--update_strength", type=float, default=0.4)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=10)
    parser.add_argument("--output_scale", type=float, default=1)
    parser.add_argument("--view_threshold", type=float, default=0.1)
    parser.add_argument("--num_viewpoints", type=int, default=1)
    parser.add_argument("--viewpoint_mode", type=str, default="predefined", choices=["predefined", "hemisphere"])
    parser.add_argument("--update_steps", type=int, default=20)
    parser.add_argument("--update_mode", type=str, default="heuristic", choices=["sequential", "heuristic", "random"])
    parser.add_argument("--blend", type=float, default=0.0)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    # 必须提供 view0 彩色图（会替换 generate/rendering/0.png 并作为 view0 的 generate_image 进行 back-project）
    parser.add_argument("--override_rendering_0", type=str, default="data/texture_images/round_bird/ipadapter_result_00.png",
                        help="Path to colored view-0 image (e.g., ipadapter_result_00.png). Used to seed the initial texture.")
    parser.add_argument("--skip_view0_new_stage", type=str2bool, default=True,
                        help="Keep True: view0 will skip diffusion and directly back-project the provided override image.")

    # IP-Adapter arguments
    parser.add_argument("--style_img", type=str, default="data/texture_images/round_bird/cat2.jpg")
    parser.add_argument("--style_img_bg_color", type=int, default=[255, 255, 255], nargs="+")
    parser.add_argument("--ip_adapter_path", type=str, default="./ip_adapter")
    parser.add_argument("--ip_adapter_strength", type=float, default=1.0)
    parser.add_argument("--ip_adapter_n_tokens", type=int, default=16)

    # ControlNet arguments
    parser.add_argument("--controlnet_cond", type=str, default="canny", choices=["canny", "depth"])
    parser.add_argument("--controlnet_strength", type=float, default=1.0)
    parser.add_argument("--use_cc_edges", type=str2bool, default=True)
    parser.add_argument("--use_depth_edges", type=str2bool, default=True)
    parser.add_argument("--use_normal_edges", type=str2bool, default=True)

    # misc
    parser.add_argument("--use_multiple_objects", action="store_true")
    parser.add_argument("--use_principle", action="store_true")
    parser.add_argument("--use_shapenet", action="store_true")
    parser.add_argument("--use_objaverse", action="store_true")
    parser.add_argument("--use_unnormalized", action="store_true")
    parser.add_argument("--add_view_to_prompt", action="store_true")
    parser.add_argument("--post_process", action="store_true")
    parser.add_argument("--smooth_mask", action="store_true")
    parser.add_argument("--force", action="store_true")
    # 保持与原 generate_texture.py 一致的开关（影响保存结构/命名）
    parser.add_argument("--no_repaint", action="store_true", help="do NOT apply repaint")
    parser.add_argument("--no_update", action="store_true", help="do NOT apply update stage")
    parser.add_argument(
        "--run_suffix",
        type=str,
        default="-inpaintall",
        help="Optional suffix appended to output_dir folder name (set '' to match generate_texture.py naming exactly).",
    )

    parser.add_argument("--tex_resolution", type=str, choices=["1k", "3k"], default="1k")
    parser.add_argument("--dist", type=float, default=0.8, help="distance to the camera from the object")
    parser.add_argument("--elev", type=float, default=0)
    parser.add_argument("--azim", type=float, default=180)

    # 让脚本“直接运行”时行为与用户命令一致（仍可用命令行覆盖 True）
    parser.set_defaults(add_view_to_prompt=True, post_process=True, use_objaverse=True, use_principle=True)
    args = parser.parse_args()

    # keep same defaults as easi-tex
    if args.tex_resolution == "3k":
        setattr(args, "render_simple_factor", 12)
        setattr(args, "fragment_k", 1)
        setattr(args, "image_size", 768)
        setattr(args, "uv_size", 3000)
    else:
        setattr(args, "render_simple_factor", 4)
        setattr(args, "fragment_k", 1)
        setattr(args, "image_size", 768)
        setattr(args, "uv_size", 1000)

    return args


def _load_ipadapter_embeds(args):
    """
    与 generate_texture.py 保持一致：style_img -> mask -> recenter_object(512,0.9) -> swap_background -> 224 -> CLIP -> Resampler
    """
    assert os.path.isfile(args.style_img)
    ip_image_encoder_path = os.path.join(args.ip_adapter_path, "image_encoder")
    ipadapter_ckpt_path = os.path.join(args.ip_adapter_path, "ip-adapter-plus_sd15.bin")
    ipadapter_model = torch.load(ipadapter_ckpt_path, map_location="cpu")

    ip_is_plus = "latents" in ipadapter_model["image_proj"]
    ip_adapter_weights = ipadapter_model.get("ip_adapter", {})
    ip_output_cross_attention_dim = ip_adapter_weights["1.to_k_ip.weight"].shape[1]
    ip_is_sdxl = ip_output_cross_attention_dim == 2048
    ip_cross_attention_dim = 1280 if ip_is_plus and ip_is_sdxl else ip_output_cross_attention_dim
    ip_heads = 20 if ip_is_sdxl and ip_is_plus else 12
    ip_num_tokens = 16 if ip_is_plus else 4

    ip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(ip_image_encoder_path).to(DEVICE, dtype=torch.float16)
    ip_clip_image_processor = CLIPImageProcessor(resample=Image.Resampling.LANCZOS)

    ip_image_proj_model = Resampler(
        dim=ip_cross_attention_dim,
        depth=4,
        dim_head=64,
        heads=ip_heads,
        num_queries=ip_num_tokens,
        embedding_dim=ip_image_encoder.config.hidden_size,
        output_dim=ip_output_cross_attention_dim,
        ff_mult=4,
    ).to(DEVICE, dtype=torch.float16)
    ip_image_proj_model.load_state_dict(ipadapter_model["image_proj"])

    img_path_split = args.style_img.rpartition(".")
    style_image = ImageUtils.load_image(args.style_img)
    style_image_mask = ImageUtils.load_image(img_path_split[0] + "_mask" + img_path_split[1] + img_path_split[2])
    style_image, style_image_mask = ImageUtils.recenter_object(style_image, style_image_mask, 512, 0.9)

    aug_style_image = ImageUtils.swap_background(style_image, style_image_mask, args.style_img_bg_color)
    aug_style_image = cv2.resize(aug_style_image, (224, 224), interpolation=cv2.INTER_AREA)
    aug_style_image = Image.fromarray(aug_style_image).convert("RGB")

    with torch.inference_mode():
        clip_image = ip_clip_image_processor(images=aug_style_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(DEVICE, dtype=torch.float16)
        clip_image_embeds = ip_image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = ip_image_proj_model(clip_image_embeds)
        negative_clip_image_embeds = ip_image_encoder(torch.zeros_like(clip_image), output_hidden_states=True).hidden_states[-2]
        negative_image_prompt_embeds = ip_image_proj_model(negative_clip_image_embeds)

    del ip_clip_image_processor
    del ip_image_encoder
    torch.cuda.empty_cache()

    num_tokens = image_prompt_embeds.shape[1]
    return image_prompt_embeds, negative_image_prompt_embeds, num_tokens, ip_adapter_weights


def _init_diffusers_inpaint_pipe(args, *, num_tokens: int, ip_adapter_weights: dict):
    """
    真·inpainting：StableDiffusionControlNetInpaintPipeline + ControlNet，并挂载 IP-Adapter attention processors。
    """
    # 选择 SD1.5 inpaint 基座（与 sd15 ip-adapter 权重兼容）
    base_inpaint_id = "runwayml/stable-diffusion-inpainting"
    if args.controlnet_cond == "depth":
        controlnet_id = "lllyasviel/control_v11f1p_sd15_depth"
    else:
        controlnet_id = "lllyasviel/control_v11p_sd15_canny"

    controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16).to(DEVICE)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        base_inpaint_id,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    # 挂载 IP-Adapter attention processors（与 test_Ipadapter.py 同思路）
    setup_ipadapter_attention_processors(
        pipe=pipe,
        num_tokens=int(num_tokens),
        device=DEVICE,
        ip_adapter_weights=ip_adapter_weights,
        ip_adapter_path=args.ip_adapter_path,
        dtype=next(pipe.unet.parameters()).dtype,
    )
    for attn_processor in pipe.unet.attn_processors.values():
        if isinstance(attn_processor, (IPAttnProcessor, IPAttnProcessor2_0)):
            attn_processor.scale = float(getattr(args, "ip_adapter_strength", 1.0))

    return pipe


def _encode_prompt_with_ip(
    pipe: StableDiffusionControlNetInpaintPipeline,
    prompt: str,
    negative_prompt: str,
    *,
    image_prompt_embeds: torch.Tensor,
    negative_image_prompt_embeds: torch.Tensor,
):
    """
    把 IP-Adapter 的 image embeds 拼接到文本 prompt embeds 后面（序列维度拼接）。
    """
    do_cfg = float(getattr(pipe, "guidance_scale", 7.5)) > 1.0  # not reliable; we compute below
    # diffusers 的 encode_prompt 需要显式传 do_classifier_free_guidance
    do_cfg = True  # 本项目基本都开 CFG（guidance_scale>=1），为安全这里按 True 处理
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        device=DEVICE,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg,
    )

    # to dtype + batch align
    to_dtype = prompt_embeds.dtype
    bsz = prompt_embeds.shape[0]
    ip_pos = image_prompt_embeds.to(to_dtype)
    ip_neg = negative_image_prompt_embeds.to(to_dtype)
    if ip_pos.shape[0] != bsz:
        ip_pos = ip_pos.repeat(bsz, 1, 1)
    if ip_neg.shape[0] != negative_prompt_embeds.shape[0]:
        ip_neg = ip_neg.repeat(negative_prompt_embeds.shape[0], 1, 1)

    prompt_embeds = torch.cat([prompt_embeds, ip_pos], dim=1)
    negative_prompt_embeds = torch.cat([negative_prompt_embeds, ip_neg], dim=1)
    return prompt_embeds, negative_prompt_embeds


def _compose_inpaint(base_image_rgb: Image.Image, gen_image_rgb: Image.Image, mask_l: Image.Image) -> Image.Image:
    """
    严格 inpainting 合成（保证 mask=0 的已知区域像素不变）：
      out = gen * mask + base * (1-mask)
    - mask: 白(255)=unknown(需要生成)，黑(0)=known(保持)
    """
    base = base_image_rgb.convert("RGB")
    gen = gen_image_rgb.convert("RGB").resize(base.size, Image.Resampling.LANCZOS)
    m = mask_l.convert("L").resize(base.size, Image.Resampling.NEAREST)
    m_np = (np.array(m).astype(np.float32) / 255.0)[..., None]
    out = np.array(gen).astype(np.float32) * m_np + np.array(base).astype(np.float32) * (1.0 - m_np)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8)).convert("RGB")


def _diffusers_controlnet_inpaint(
    pipe: StableDiffusionControlNetInpaintPipeline,
    *,
    base_image: Image.Image,
    mask_image: Image.Image,
    control_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    controlnet_conditioning_scale: float,
    image_prompt_embeds: torch.Tensor,
    negative_image_prompt_embeds: torch.Tensor,
    strict_compose: bool = True,
) -> Image.Image:
    prompt_embeds, negative_prompt_embeds = _encode_prompt_with_ip(
        pipe,
        prompt,
        negative_prompt,
        image_prompt_embeds=image_prompt_embeds,
        negative_image_prompt_embeds=negative_image_prompt_embeds,
    )
    gen = torch.Generator(device=DEVICE).manual_seed(int(seed))
    out = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        image=base_image.convert("RGB").resize((pipe.unet.config.sample_size * 8, pipe.unet.config.sample_size * 8), Image.Resampling.LANCZOS),
        mask_image=mask_image.convert("L").resize((pipe.unet.config.sample_size * 8, pipe.unet.config.sample_size * 8), Image.Resampling.NEAREST),
        control_image=control_image.convert("RGB").resize((pipe.unet.config.sample_size * 8, pipe.unet.config.sample_size * 8), Image.Resampling.NEAREST),
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        strength=float(strength),
        generator=gen,
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        output_type="pil",
    ).images[0]
    if strict_compose:
        out = _compose_inpaint(
            base_image_rgb=base_image.convert("RGB").resize(out.size, Image.Resampling.LANCZOS),
            gen_image_rgb=out,
            mask_l=mask_image,
        )
    return out


if __name__ == "__main__":
    args = init_args()
    assert len(args.style_img_bg_color) == 3

    if not os.path.isfile(args.override_rendering_0):
        raise FileNotFoundError(f"--override_rendering_0 not found: {args.override_rendering_0}")

    # output_dir 命名规则：严格对齐 generate_texture.py（仅可选追加 run_suffix）
    output_dir = os.path.join(
        args.output_dir,
        args.controlnet_cond,
        f"{args.input_dir.split('/')[-1]}-{args.style_img.split('/')[-2]}",
        "{}-{}-{}-{}-{}-{}-{}-{}-{}".format(
            str(args.seed),
            "ip" + str(args.ip_adapter_strength),
            "cn" + str(args.controlnet_strength),
            "dist" + str(args.dist),
            "gs" + str(args.guidance_scale),
            args.viewpoint_mode[0] + str(args.num_viewpoints),
            args.update_mode[0] + str(args.update_steps),
            "us" + str(args.update_strength),
            "vt" + str(args.view_threshold),
        ),
    )
    if args.no_repaint:
        output_dir += "-norepaint"
    if args.no_update:
        output_dir += "-noupdate"
    if getattr(args, "run_suffix", ""):
        output_dir += str(args.run_suffix)
    os.makedirs(output_dir, exist_ok=True)
    print("=> OUTPUT_DIR:", output_dir)

    edge_selector = {
        "use_cc_edges": args.use_cc_edges,
        "use_depth_edges": args.use_depth_edges,
        "use_normal_edges": args.use_normal_edges,
    }

    # mesh init (same as easi-tex)
    mesh, _, faces, aux, principle_directions, mesh_center, mesh_scale, edge_mesh = init_mesh(
        os.path.join(args.input_dir, args.obj_file),
        os.path.join(output_dir, f"{args.obj_file.rsplit('.')[0]}_xatlas.{args.obj_file.rsplit('.')[1]}"),
        DEVICE,
    )

    # dummy texture start
    init_texture = Image.open("./samples/textures/dummy.png").convert("RGB").resize((args.uv_size, args.uv_size))
    if args.use_multiple_objects:
        new_verts_uvs, init_texture = adjust_uv_map(faces, aux, init_texture, args.uv_size)
    else:
        new_verts_uvs = aux.verts_uvs

    mesh.textures = TexturesUV(
        maps=transforms.ToTensor()(init_texture)[None, ...].permute(0, 2, 3, 1).to(DEVICE),
        faces_uvs=faces.textures_idx[None, ...],
        verts_uvs=new_verts_uvs[None, ...],
    )

    exist_texture = torch.from_numpy(np.zeros([args.uv_size, args.uv_size]).astype(np.float32)).to(DEVICE)

    (dist_list, elev_list, azim_list, sector_list, view_punishments) = init_viewpoints(
        args.viewpoint_mode, args.num_viewpoints, args.dist, args.elev, principle_directions,
        use_principle=True,
        use_shapenet=args.use_shapenet,
        use_objaverse=args.use_objaverse,
    )

    save_args(args, output_dir)

    # ===== diffusers: 真 inpainting pipeline（ControlNet + IP-Adapter）=====
    image_prompt_embeds, negative_image_prompt_embeds, ip_num_tokens, ip_adapter_weights = _load_ipadapter_embeds(args)
    inpaint_pipe = _init_diffusers_inpaint_pipe(args, num_tokens=ip_num_tokens, ip_adapter_weights=ip_adapter_weights)

    # dirs (generate)
    generate_dir = os.path.join(output_dir, "generate")
    os.makedirs(generate_dir, exist_ok=True)
    init_image_dir = os.path.join(generate_dir, "rendering")
    normal_map_dir = os.path.join(generate_dir, "normal")
    mask_image_dir = os.path.join(generate_dir, "mask")
    depth_map_dir = os.path.join(generate_dir, "depth" if args.controlnet_cond == "depth" else "edges")
    similarity_map_dir = os.path.join(generate_dir, "similarity")
    inpainted_image_dir = os.path.join(generate_dir, "inpainted")
    mesh_dir = os.path.join(generate_dir, "mesh")
    interm_dir = os.path.join(generate_dir, "intermediate")
    for d in [init_image_dir, normal_map_dir, mask_image_dir, depth_map_dir, similarity_map_dir, inpainted_image_dir, mesh_dir, interm_dir]:
        os.makedirs(d, exist_ok=True)

    # principle views
    NUM_PRINCIPLE = 10 if args.use_shapenet or args.use_objaverse else 6
    pre_dist_list = dist_list[:NUM_PRINCIPLE]
    pre_elev_list = elev_list[:NUM_PRINCIPLE]
    pre_azim_list = azim_list[:NUM_PRINCIPLE]
    pre_sector_list = sector_list[:NUM_PRINCIPLE]
    pre_view_punishments = view_punishments[:NUM_PRINCIPLE]

    pre_similarity_texture_cache = build_similarity_texture_cache_for_all_views(
        mesh, edge_mesh, faces, new_verts_uvs,
        pre_dist_list, pre_elev_list, pre_azim_list,
        args.image_size, args.image_size * args.render_simple_factor, args.uv_size, args.fragment_k,
        DEVICE, controlnet_cond=args.controlnet_cond, edge_selector=edge_selector,
    )

    print("=> start generating texture (inpaint-all-views)...")
    start_time = time.time()

    for view_idx in range(NUM_PRINCIPLE):
        print(f"=> processing view {view_idx}...")
        dist, elev, azim, sector = pre_dist_list[view_idx], pre_elev_list[view_idx], pre_azim_list[view_idx], pre_sector_list[view_idx]
        prompt = f" the {sector} view of {args.prompt}" if args.add_view_to_prompt else args.prompt
        print(f"=> generating image for prompt: {prompt}...")

        # render and build masks on CURRENT textured mesh (this is the key for consistency)
        (
            view_score,
            renderer, cameras, camera_pos, fragments,
            init_image, normal_map, depth_map,
            init_images_tensor, normal_maps_tensor, depth_maps_tensor, similarity_tensor,
            keep_mask_image, update_mask_image, generate_mask_image,
            keep_mask_tensor, update_mask_tensor, generate_mask_tensor, all_mask_tensor, quad_mask_tensor,
        ) = render_one_view_and_build_masks(
            dist, elev, azim,
            view_idx, view_idx, pre_view_punishments,
            pre_similarity_texture_cache, exist_texture,
            mesh, edge_mesh, faces, new_verts_uvs,
            args.image_size, args.fragment_k,
            init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir,
            DEVICE, controlnet_cond=args.controlnet_cond, edge_selector=edge_selector, save_intermediate=True,
            smooth_mask=args.smooth_mask, view_threshold=args.view_threshold,
        )

        # 1.2 generate missing region (true inpainting)
        # NOTE: keep the same semantics as generate_texture.py when --no_repaint is set
        if args.no_repaint and view_idx != 0:
            actual_generate_mask_image = Image.fromarray((np.ones_like(np.array(generate_mask_image)) * 255.).astype(np.uint8))
        else:
            actual_generate_mask_image = generate_mask_image

        # view0: seed texture from provided colored image (skip diffusion)
        if view_idx == 0:
            override_img = Image.open(args.override_rendering_0).convert("RGB").resize(
                (args.image_size, args.image_size), Image.Resampling.LANCZOS
            )
            init_image = override_img
            init_image.save(os.path.join(init_image_dir, "0.png"))
            generate_image = init_image
            generate_image_before = init_image
            generate_image_after = init_image
            print("=> view0: use override image and skip diffusion (texture seed).")
        else:
            # view>0: 真·inpainting（StableDiffusionControlNetInpaintPipeline）
            # 1) control_image：与原 easi-tex 保持一致，使用 depth_maps_tensor（depth 或 edges），并转成 3 通道
            ctrl = depth_maps_tensor.permute(1, 2, 0).repeat(1, 1, 3).detach().cpu().numpy()
            ctrl = np.clip(ctrl, 0, 255).astype(np.uint8)
            control_image = Image.fromarray(ctrl).convert("RGB")

            # 2) mask_image：generate_mask_image 白=需要 inpaint，黑=保持
            mask_image = generate_mask_image.convert("L")
            base_image = init_image.convert("RGB")

            # 3) 真 inpaint + 严格合成（保证非 mask 区域不变）
            generate_image = _diffusers_controlnet_inpaint(
                inpaint_pipe,
                base_image=base_image.resize((args.image_size, args.image_size), Image.Resampling.LANCZOS),
                mask_image=actual_generate_mask_image.convert("L").resize((args.image_size, args.image_size), Image.Resampling.NEAREST),
                control_image=control_image.resize((args.image_size, args.image_size), Image.Resampling.NEAREST),
                prompt=prompt,
                negative_prompt=args.n_prompt,
                strength=float(args.inpaint_strength),
                num_inference_steps=int(args.ddim_steps),
                guidance_scale=float(args.guidance_scale),
                seed=int(args.seed),
                controlnet_conditioning_scale=float(args.controlnet_strength),
                image_prompt_embeds=image_prompt_embeds,
                negative_image_prompt_embeds=negative_image_prompt_embeds,
                strict_compose=True,
            ).resize((args.image_size, args.image_size), Image.Resampling.LANCZOS)
            # before/after：这里没有“keep region blend”步骤，先用同一张占位保持目录结构
            generate_image_before = generate_image
            generate_image_after = generate_image

        generate_image.save(os.path.join(inpainted_image_dir, f"{view_idx}.png"))
        generate_image_before.save(os.path.join(inpainted_image_dir, f"{view_idx}_before.png"))
        generate_image_after.save(os.path.join(inpainted_image_dir, f"{view_idx}_after.png"))

        # back-project (projection mask = generate mask)
        init_texture, project_mask_image, exist_texture = backproject_from_image(
            mesh, faces, new_verts_uvs, cameras,
            generate_image, generate_mask_image, generate_mask_image, init_texture, exist_texture,
            args.image_size * args.render_simple_factor, args.uv_size, args.fragment_k,
            DEVICE,
        )
        project_mask_image.save(os.path.join(mask_image_dir, f"{view_idx}_project.png"))

        # update mesh textures for next view
        mesh.textures = TexturesUV(
            maps=transforms.ToTensor()(init_texture)[None, ...].permute(0, 2, 3, 1).to(DEVICE),
            faces_uvs=faces.textures_idx[None, ...],
            verts_uvs=new_verts_uvs[None, ...],
        )

        # save backprojected OBJ (optional but keeps structure consistent)
        save_backproject_obj(
            mesh_dir, f"{view_idx}.obj",
            mesh_scale * mesh.verts_packed() + mesh_center if args.use_unnormalized else mesh.verts_packed(),
            faces.verts_idx, new_verts_uvs, faces.textures_idx, init_texture,
            DEVICE,
        )

        # save intermediate view (render current textured mesh)
        inter_images_tensor, *_ = render(
            mesh, edge_mesh, renderer, controlnet_cond=args.controlnet_cond, edge_selector=edge_selector, camera_pos=camera_pos
        )
        inter_image = inter_images_tensor[0].cpu()
        inter_image = inter_image.permute(2, 0, 1)
        inter_image = transforms.ToPILImage()(inter_image).convert("RGB")
        inter_image.save(os.path.join(interm_dir, f"{view_idx}.png"))

        # save texture mask（与 generate_texture.py 一致）
        exist_texture_image = exist_texture * 255.
        exist_texture_image = Image.fromarray(exist_texture_image.cpu().numpy().astype(np.uint8)).convert("L")
        exist_texture_image.save(os.path.join(mesh_dir, "{}_texture_mask.png".format(view_idx)))

    print("=> total generate time: {} s".format(time.time() - start_time))
    visualize_principle_viewpoints(output_dir, pre_dist_list, pre_elev_list, pre_azim_list)

    # update stage：保持 generate_texture.py 的保存结构（即使 --no_update 也会生成 update/ 目录并保存中间结果）
    if args.update_steps > 0:
        update_dir = os.path.join(output_dir, "update")
        os.makedirs(update_dir, exist_ok=True)
        init_image_dir = os.path.join(update_dir, "rendering")
        normal_map_dir = os.path.join(update_dir, "normal")
        mask_image_dir = os.path.join(update_dir, "mask")
        depth_map_dir = os.path.join(update_dir, "depth" if args.controlnet_cond == "depth" else "edges")
        similarity_map_dir = os.path.join(update_dir, "similarity")
        inpainted_image_dir = os.path.join(update_dir, "inpainted")
        mesh_dir = os.path.join(update_dir, "mesh")
        interm_dir = os.path.join(update_dir, "intermediate")
        for d in [init_image_dir, normal_map_dir, mask_image_dir, depth_map_dir, similarity_map_dir, inpainted_image_dir, mesh_dir, interm_dir]:
            os.makedirs(d, exist_ok=True)

        dist_list2 = dist_list[NUM_PRINCIPLE:]
        elev_list2 = elev_list[NUM_PRINCIPLE:]
        azim_list2 = azim_list[NUM_PRINCIPLE:]
        sector_list2 = sector_list[NUM_PRINCIPLE:]
        view_punishments2 = view_punishments[NUM_PRINCIPLE:]

        similarity_texture_cache = build_similarity_texture_cache_for_all_views(
            mesh, edge_mesh, faces, new_verts_uvs,
            dist_list2, elev_list2, azim_list2,
            args.image_size, args.image_size * args.render_simple_factor, args.uv_size, args.fragment_k,
            DEVICE, controlnet_cond=args.controlnet_cond, edge_selector=edge_selector,
        )
        selected_view_ids = []

        print("=> start updating...")
        start_time = time.time()
        for view_idx in range(args.update_steps):
            dist, elev, azim, sector, selected_view_ids, view_punishments2 = select_viewpoint(
                selected_view_ids, view_punishments2,
                args.update_mode, dist_list2, elev_list2, azim_list2, sector_list2, view_idx,
                similarity_texture_cache, exist_texture,
                mesh, edge_mesh, faces, new_verts_uvs,
                args.image_size, args.fragment_k,
                init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir,
                DEVICE, args.controlnet_cond, edge_selector, False,
            )

            (
                view_score,
                renderer, cameras, camera_pos, fragments,
                init_image, normal_map, depth_map,
                init_images_tensor, normal_maps_tensor, depth_maps_tensor, similarity_tensor,
                old_mask_image, update_mask_image, generate_mask_image,
                old_mask_tensor, update_mask_tensor, generate_mask_tensor, all_mask_tensor, quad_mask_tensor,
            ) = render_one_view_and_build_masks(
                dist, elev, azim,
                selected_view_ids[-1], view_idx, view_punishments2,
                similarity_texture_cache, exist_texture,
                mesh, edge_mesh, faces, new_verts_uvs,
                args.image_size, args.fragment_k,
                init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir,
                DEVICE, controlnet_cond=args.controlnet_cond, edge_selector=edge_selector, save_intermediate=True,
                smooth_mask=args.smooth_mask, view_threshold=args.view_threshold,
            )

            prompt = f" the {sector} view of {args.prompt}" if args.add_view_to_prompt else args.prompt
            print("=> updating image for prompt: {}...".format(prompt))

            if (not args.no_update) and update_mask_tensor.sum() > 0 and update_mask_tensor.sum() / (all_mask_tensor.sum()) > 0.05:
                ctrl = depth_maps_tensor.permute(1, 2, 0).repeat(1, 1, 3).detach().cpu().numpy()
                ctrl = np.clip(ctrl, 0, 255).astype(np.uint8)
                control_image = Image.fromarray(ctrl).convert("RGB")
                update_image = _diffusers_controlnet_inpaint(
                    inpaint_pipe,
                    base_image=init_image.convert("RGB").resize((args.image_size, args.image_size), Image.Resampling.LANCZOS),
                    mask_image=update_mask_image.convert("L").resize((args.image_size, args.image_size), Image.Resampling.NEAREST),
                    control_image=control_image.resize((args.image_size, args.image_size), Image.Resampling.NEAREST),
                    prompt=prompt,
                    negative_prompt=args.n_prompt,
                    strength=float(args.update_strength),
                    num_inference_steps=int(args.ddim_steps),
                    guidance_scale=float(args.guidance_scale),
                    seed=int(args.seed),
                    controlnet_conditioning_scale=float(args.controlnet_strength),
                    image_prompt_embeds=image_prompt_embeds,
                    negative_image_prompt_embeds=negative_image_prompt_embeds,
                    strict_compose=True,
                ).resize((args.image_size, args.image_size), Image.Resampling.LANCZOS)
                update_image_before = update_image
                update_image_after = update_image
                update_image.save(os.path.join(inpainted_image_dir, f"{view_idx}.png"))
                update_image_before.save(os.path.join(inpainted_image_dir, f"{view_idx}_before.png"))
                update_image_after.save(os.path.join(inpainted_image_dir, f"{view_idx}_after.png"))
            else:
                update_image = init_image

            init_texture, project_mask_image, exist_texture = backproject_from_image(
                mesh, faces, new_verts_uvs, cameras,
                update_image, update_mask_image, update_mask_image, init_texture, exist_texture,
                args.image_size * args.render_simple_factor, args.uv_size, args.fragment_k,
                DEVICE,
            )
            project_mask_image.save(os.path.join(mask_image_dir, f"{view_idx}_project.png"))

            mesh.textures = TexturesUV(
                maps=transforms.ToTensor()(init_texture)[None, ...].permute(0, 2, 3, 1).to(DEVICE),
                faces_uvs=faces.textures_idx[None, ...],
                verts_uvs=new_verts_uvs[None, ...],
            )

            save_backproject_obj(
                mesh_dir, f"{view_idx}.obj",
                mesh_scale * mesh.verts_packed() + mesh_center if args.use_unnormalized else mesh.verts_packed(),
                faces.verts_idx, new_verts_uvs, faces.textures_idx, init_texture,
                DEVICE,
            )

            inter_images_tensor, *_ = render(
                mesh, edge_mesh, renderer, controlnet_cond=args.controlnet_cond, edge_selector=edge_selector, camera_pos=camera_pos
            )
            inter_image = inter_images_tensor[0].cpu().permute(2, 0, 1)
            transforms.ToPILImage()(inter_image).convert("RGB").save(os.path.join(interm_dir, f"{view_idx}.png"))

            # save texture mask（与 generate_texture.py 一致）
            exist_texture_image = exist_texture * 255.
            exist_texture_image = Image.fromarray(exist_texture_image.cpu().numpy().astype(np.uint8)).convert("L")
            exist_texture_image.save(os.path.join(mesh_dir, "{}_texture_mask.png".format(view_idx)))

        print("=> total update time: {} s".format(time.time() - start_time))

        # post-process
        if args.post_process:
            inpainting = get_inpainting(DEVICE)
            post_texture = apply_inpainting_postprocess(
                inpainting,
                init_texture, 1 - exist_texture[None, :, :, None], "", args.uv_size, args.uv_size, DEVICE,
            )
            save_backproject_obj(
                mesh_dir, f"{view_idx}_post.obj",
                mesh_scale * mesh.verts_packed() + mesh_center if args.use_unnormalized else mesh.verts_packed(),
                faces.verts_idx, new_verts_uvs, faces.textures_idx, post_texture,
                DEVICE,
            )

        save_viewpoints(args, output_dir, dist_list2, elev_list2, azim_list2, selected_view_ids)
        visualize_refinement_viewpoints(output_dir, selected_view_ids, dist_list2, elev_list2, azim_list2)


