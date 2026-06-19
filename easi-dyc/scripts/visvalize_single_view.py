import os
import argparse
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from pytorch3d.renderer import TexturesUV
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

sys.path.append(".")

from lib.mesh_helper import init_mesh, adjust_uv_map
from lib.projection_helper import (
    render_one_view_and_build_masks,
    build_similarity_texture_cache_for_all_views,
)
from lib.diffusion_helper import (
    get_controlnet_depth,
    apply_controlnet_depth,
    get_inpainting,
    apply_inpainting_postprocess,
)
from lib.dataloader_utils import ImageUtils
from ip_adapter.resampler import Resampler


if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(DEVICE)
else:
    print("no gpu avaiable")
    sys.exit(1)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize init/generate/update images for a single view."
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--obj_file", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./single_view_outputs",
        help="Directory to store visualization images.",
    )
    parser.add_argument(
        "--style_img",
        type=str,
        default="",
        help="Reference style image (same format as generate_texture.py).",
    )
    parser.add_argument(
        "--style_img_bg_color",
        type=int,
        nargs="+",
        default=[255, 255, 255],
        help="Background color used when recentering the style image.",
    )
    parser.add_argument("--ip_adapter_path", type=str, default="./ip_adapter")
    parser.add_argument("--ip_adapter_strength", type=float, default=0.6)
    parser.add_argument("--ip_adapter_n_tokens", type=int, default=16)

    parser.add_argument(
        "--controlnet_cond",
        type=str,
        default="depth",
        choices=["depth", "canny"],
        help="Condition type for ControlNet.",
    )
    parser.add_argument("--controlnet_strength", type=float, default=1.0)
    parser.add_argument("--use_cc_edges", type=str2bool, default=True)
    parser.add_argument("--use_depth_edges", type=str2bool, default=True)
    parser.add_argument("--use_normal_edges", type=str2bool, default=True)

    parser.add_argument("--new_strength", type=float, default=1.0)
    parser.add_argument("--update_strength", type=float, default=0.5)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=10.0)
    parser.add_argument("--a_prompt", type=str, default="best quality, high quality")
    parser.add_argument("--n_prompt", type=str, default="deformed, low quality")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--blend", type=float, default=0.5)

    parser.add_argument("--no_repaint", action="store_true")
    parser.add_argument("--no_update", action="store_true")
    parser.add_argument(
        "--enable_inpaint",
        action="store_true",
        help="启用额外的 Inpainting 后处理（默认使用更新/生成掩码）",
    )

    parser.add_argument("--tex_resolution", type=str, choices=["1k", "3k"], default="1k")
    parser.add_argument("--smooth_mask", action="store_true")
    parser.add_argument("--view_threshold", type=float, default=0.1)
    parser.add_argument("--use_multiple_objects", action="store_true")

    parser.add_argument("--dist", type=float, default=1.0, help="Camera distance.")
    parser.add_argument("--elev", type=float, default=0.0, help="Camera elevation.")
    parser.add_argument(
        "--azim",
        type=float,
        default=0.0,
        help="Camera azimuth (90 ~ side view, 0/180 ~ front/back).",
    )

    parser.add_argument("--render_simple_factor", type=int, default=None)
    parser.add_argument("--fragment_k", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--uv_size", type=int, default=None)

    args = parser.parse_args()

    if args.tex_resolution == "3k":
        args.render_simple_factor = 12
        args.fragment_k = 1
        args.image_size = 768
        args.uv_size = 3000
    else:
        args.render_simple_factor = 4
        args.fragment_k = 1
        args.image_size = 768
        args.uv_size = 1000

    if len(args.style_img_bg_color) != 3:
        parser.error("--style_img_bg_color must have 3 values (RGB).")
    bg = np.array(args.style_img_bg_color)
    if np.any((bg < 0) | (bg > 255)):
        parser.error("--style_img_bg_color values must be within [0, 255].")

    return args


def ensure_dirs(base_dir):
    subdirs = ["rendering", "normal", "depth", "mask", "similarity"]
    paths = {}
    for name in subdirs:
        path = os.path.join(base_dir, name)
        os.makedirs(path, exist_ok=True)
        paths[name] = path
    return paths


def load_style_embeddings(args):
    if args.style_img == "":
        return None, None

    assert os.path.isfile(args.style_img), f"style image {args.style_img} not found"

    ip_image_encoder_path = os.path.join(args.ip_adapter_path, "image_encoder")
    ipadapter_ckpt_path = os.path.join(args.ip_adapter_path, "ip-adapter-plus_sd15.bin")
    ipadapter_model = torch.load(ipadapter_ckpt_path, map_location=DEVICE)

    ip_is_plus = "latents" in ipadapter_model["image_proj"]
    ip_output_cross_attention_dim = ipadapter_model["ip_adapter"]["1.to_k_ip.weight"].shape[
        1
    ]
    ip_is_sdxl = ip_output_cross_attention_dim == 2048
    ip_cross_attention_dim = (
        1280 if ip_is_plus and ip_is_sdxl else ip_output_cross_attention_dim
    )
    ip_heads = 20 if ip_is_sdxl and ip_is_plus else 12
    ip_num_tokens = 16 if ip_is_plus else 4

    ip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        ip_image_encoder_path
    ).to(DEVICE, dtype=torch.float16)
    ip_clip_image_processor = CLIPImageProcessor(
        resample=Image.Resampling.LANCZOS
    )

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
    style_image_mask = ImageUtils.load_image(
        img_path_split[0] + "_mask" + img_path_split[1] + img_path_split[2]
    )
    style_image, style_image_mask = ImageUtils.recenter_object(
        style_image, style_image_mask, 512, 0.9
    )
    aug_style_image = ImageUtils.swap_background(
        style_image, style_image_mask, args.style_img_bg_color
    )
    aug_style_image = Image.fromarray(
        cv2.resize(aug_style_image, (224, 224), interpolation=cv2.INTER_AREA)
    ).convert("RGB")

    with torch.inference_mode():
        clip_image = ip_clip_image_processor(images=aug_style_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(DEVICE, dtype=torch.float16)

        clip_image_embeds = ip_image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = ip_image_proj_model(clip_image_embeds)

        negative_clip_image_embeds = ip_image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        negative_image_prompt_embeds = ip_image_proj_model(negative_clip_image_embeds)

    del ip_clip_image_processor
    del ip_image_encoder
    torch.cuda.empty_cache()

    return image_prompt_embeds, negative_image_prompt_embeds


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dir_map = ensure_dirs(args.output_dir)

    edge_selector = {
        "use_cc_edges": args.use_cc_edges,
        "use_depth_edges": args.use_depth_edges,
        "use_normal_edges": args.use_normal_edges,
    }

    mesh, _, faces, aux, principle_directions, _, _, edge_mesh = init_mesh(
        os.path.join(args.input_dir, args.obj_file),
        os.path.join(
            args.output_dir, f"{args.obj_file.rsplit('.')[0]}_xatlas.{args.obj_file.rsplit('.')[1]}"
        ),
        DEVICE,
    )

    init_texture = Image.open("./samples/textures/dummy.png").convert("RGB").resize(
        (args.uv_size, args.uv_size)
    )
    if args.use_multiple_objects:
        new_verts_uvs, init_texture = adjust_uv_map(
            faces, aux, init_texture, args.uv_size
        )
    else:
        new_verts_uvs = aux.verts_uvs

    mesh.textures = TexturesUV(
        maps=transforms.ToTensor()(init_texture)[None, ...]
        .permute(0, 2, 3, 1)
        .to(DEVICE),
        faces_uvs=faces.textures_idx[None, ...],
        verts_uvs=new_verts_uvs[None, ...],
    )

    exist_texture = torch.zeros([args.uv_size, args.uv_size], dtype=torch.float32).to(
        DEVICE
    )

    dist_list = [args.dist]
    elev_list = [args.elev]
    azim_list = [args.azim]
    sector_list = ["custom"]
    view_punishments = [0.0]

    similarity_cache = build_similarity_texture_cache_for_all_views(
        mesh,
        edge_mesh,
        faces,
        new_verts_uvs,
        dist_list,
        elev_list,
        azim_list,
        args.image_size,
        args.image_size * args.render_simple_factor,
        args.uv_size,
        args.fragment_k,
        DEVICE,
        controlnet_cond=args.controlnet_cond,
        edge_selector=edge_selector,
    )

    controlnet, ddim_sampler = get_controlnet_depth(
        controlnet_cond=args.controlnet_cond,
        controlnet_strength=args.controlnet_strength,
        ip_adapter_strength=args.ip_adapter_strength,
        ip_adapter_path=args.ip_adapter_path,
        ip_adapter_n_tokens=args.ip_adapter_n_tokens,
    )

    image_prompt_embeds, negative_image_prompt_embeds = load_style_embeddings(args)

    (
        view_score,
        renderer,
        cameras,
        camera_pos,
        fragments,
        init_image,
        normal_map,
        depth_map,
        init_images_tensor,
        normal_maps_tensor,
        depth_maps_tensor,
        similarity_tensor,
        keep_mask_image,
        update_mask_image,
        generate_mask_image,
        keep_mask_tensor,
        update_mask_tensor,
        generate_mask_tensor,
        all_mask_tensor,
        quad_mask_tensor,
    ) = render_one_view_and_build_masks(
        args.dist,
        args.elev,
        args.azim,
        0,
        0,
        view_punishments,
        similarity_cache,
        exist_texture,
        mesh,
        edge_mesh,
        faces,
        new_verts_uvs,
        args.image_size,
        args.fragment_k,
        dir_map["rendering"],
        dir_map["mask"],
        dir_map["normal"],
        dir_map["depth"],
        dir_map["similarity"],
        DEVICE,
        controlnet_cond=args.controlnet_cond,
        edge_selector=edge_selector,
        save_intermediate=True,
        smooth_mask=args.smooth_mask,
        view_threshold=args.view_threshold,
    )

    init_path = os.path.join(args.output_dir, "init_image.png")
    init_image.save(init_path)
    print(f"=> saved init image to {init_path}")

    # 保存可视化/调试用的掩码与张量
    keep_mask_image.save(os.path.join(args.output_dir, "keep_mask.png"))
    update_mask_image.save(os.path.join(args.output_dir, "update_mask.png"))
    generate_mask_image.save(os.path.join(args.output_dir, "generate_mask.png"))
    np.save(
        os.path.join(args.output_dir, "keep_mask_tensor.npy"),
        keep_mask_tensor.detach().cpu().numpy(),
    )
    np.save(
        os.path.join(args.output_dir, "update_mask_tensor.npy"),
        update_mask_tensor.detach().cpu().numpy(),
    )
    np.save(
        os.path.join(args.output_dir, "generate_mask_tensor.npy"),
        generate_mask_tensor.detach().cpu().numpy(),
    )
    np.save(
        os.path.join(args.output_dir, "all_mask_tensor.npy"),
        all_mask_tensor.detach().cpu().numpy(),
    )
    np.save(
        os.path.join(args.output_dir, "quad_mask_tensor.npy"),
        quad_mask_tensor.detach().cpu().numpy(),
    )

    # 根据 controlnet_cond 选择条件图（depth 或 canny）
    depth_np = (
        depth_maps_tensor.permute(1, 2, 0)
        .repeat(1, 1, 3)
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    if args.controlnet_cond == "depth":
        cond_np = depth_np
    elif args.controlnet_cond == "canny":
        init_np = np.array(init_image.convert("RGB"))
        edges = cv2.Canny(cv2.cvtColor(init_np, cv2.COLOR_RGB2GRAY), 100, 200)
        cond_np = np.stack([edges] * 3, axis=-1).astype(np.float32)
    else:
        raise ValueError(f"Unknown controlnet_cond: {args.controlnet_cond}")

    if args.no_repaint:
        actual_generate_mask_image = Image.fromarray(
            (np.ones_like(np.array(generate_mask_image)) * 255).astype(np.uint8)
        )
    else:
        actual_generate_mask_image = generate_mask_image

    generate_image, generate_before, generate_after = apply_controlnet_depth(
        controlnet,
        ddim_sampler,
        init_image.convert("RGBA"),
        args.prompt,
        args.new_strength,
        args.ddim_steps,
        actual_generate_mask_image,
        keep_mask_image,
        cond_np,
        args.a_prompt,
        args.n_prompt,
        args.guidance_scale,
        args.seed,
        args.eta,
        1,
        DEVICE,
        args.blend,
        pos_img_prompt_embeds=image_prompt_embeds,
        neg_img_prompt_embeds=negative_image_prompt_embeds,
    )

    generate_image.save(os.path.join(args.output_dir, "generate_image.png"))
    generate_before.save(os.path.join(args.output_dir, "generate_before.png"))
    generate_after.save(os.path.join(args.output_dir, "generate_after.png"))

    total_pixels = max(float(all_mask_tensor.sum().item()), 1.0)
    update_ratio = float(update_mask_tensor.sum().item()) / total_pixels
    did_update = False
    if (
        not args.no_update
        and update_mask_tensor.sum().item() > 0
        and update_ratio > 0.05
    ):
        print(f"=> update stage triggered (ratio={update_ratio:.3f})")
        diffused_image, diff_before, diff_after = apply_controlnet_depth(
            controlnet,
            ddim_sampler,
            init_image.convert("RGBA"),
            args.prompt,
            args.update_strength,
            args.ddim_steps,
            update_mask_image,
            keep_mask_image,
            cond_np,
            args.a_prompt,
            args.n_prompt,
            args.guidance_scale,
            args.seed,
            args.eta,
            1,
            DEVICE,
            args.blend,
            pos_img_prompt_embeds=image_prompt_embeds,
            neg_img_prompt_embeds=negative_image_prompt_embeds,
        )
        diffused_image.save(os.path.join(args.output_dir, "update_image.png"))
        diff_before.save(os.path.join(args.output_dir, "update_before.png"))
        diff_after.save(os.path.join(args.output_dir, "update_after.png"))
        did_update = True
    else:
        print("=> update stage skipped (mask too small or disabled)")

    # optional inpainting postprocess
    if args.enable_inpaint and (did_update or generate_mask_tensor.sum().item() > 0):
        print("=> running inpainting postprocess...")
        inpaint_model = get_inpainting(DEVICE)
        # choose mask: prefer update mask if used, else generate mask
        mask_src = (
            update_mask_tensor if did_update and update_mask_tensor.sum() > 0 else generate_mask_tensor
        )
        # 统一形状到 [1, H, W, 1]，浮点 0/1，放到 DEVICE
        mask_t = mask_src
        if not torch.is_tensor(mask_t):
            mask_t = torch.as_tensor(mask_t)
        if mask_t.ndim == 3 and mask_t.shape[0] in (1, 3):  # [C,H,W] -> [H,W,C]
            mask_t = mask_t.permute(1, 2, 0)
        if mask_t.ndim == 3 and mask_t.shape[-1] > 1:  # 只取单通道
            mask_t = mask_t[..., :1]
        if mask_t.ndim == 2:
            mask_t = mask_t[..., None]
        mask_t = (mask_t > 0.5).float().unsqueeze(0).to(DEVICE)  # [1,H,W,1]

        base_image = diffused_image if did_update else generate_image
        inpainted_image = apply_inpainting_postprocess(
            inpaint_model,
            base_image,
            mask_t,
            args.prompt,
            base_image.height,
            base_image.width,
            DEVICE,
        )
        inpaint_path = os.path.join(args.output_dir, "inpaint_image.png")
        inpainted_image.save(inpaint_path)
        print(f"=> saved inpaint image to {inpaint_path}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv += [
            "--input_dir",
            "data/meshes/cow",
            "--obj_file",
            "cow_fr-z_up-y.obj",
            "--prompt",
            "photorealistic cow texture",
            "--style_img",
            "data/texture_images/round_bird/round_bird_4.jpg",
            "--style_img_bg_color",
            "255",
            "255",
            "255",
            "--ip_adapter_path",
            "./ip_adapter",
            "--output_dir",
            "./single_view_debug",
            "--dist",
            "1.0",
            "--elev",
            "0.0",
            "--azim",
            "0",
            "--controlnet_cond",
            "canny",
            "--enable_inpaint",
        ]
    main()

