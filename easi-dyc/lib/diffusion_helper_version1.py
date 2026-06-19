import os
import torch

import cv2
import numpy as np

from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# Stable Diffusion 2
from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline, 
    EulerDiscreteScheduler,
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetInpaintPipeline,
)

# customized
import sys
# Ensure repo root is on sys.path for local imports regardless of CWD.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from models.ControlNet.gradio_depth2image import init_model, process
from ip_adapter.ip_adapter import setup_ipadapter_attention_processors
from ip_adapter.attention_processor import IPAttnProcessor, IPAttnProcessor2_0
from lib.reference_attention import ensure_ref_unet_wrapped

from lib.pipeline_controlnet_inpaint_refattn import (
    StableDiffusionControlNetInpaintRefAttnPipeline,
)



def get_controlnet_depth(**kwargs):
    print("=> initializing ControlNet Depth...")
    model, ddim_sampler = init_model(**kwargs)

    return model, ddim_sampler


def _pick_local_or_hf_id(local_dir: str, hf_id: str):
    if local_dir and os.path.isdir(local_dir):
        return local_dir, True
    return hf_id, False


def _hf_cache_snapshot_dir(model_cache_dir: str) -> str:
    """
    给定 HF hub 的模型缓存目录（例如 /data/23009960/hub/models--runwayml--stable-diffusion-inpainting），
    自动解析 refs/main -> snapshots/<hash>，返回可用于 diffusers.from_pretrained 的本地目录。
    若无法解析则返回空字符串。
    """
    try:
        refs_main = os.path.join(model_cache_dir, "refs", "main")
        if os.path.isfile(refs_main):
            with open(refs_main, "r", encoding="utf-8") as f:
                rev = f.read().strip()
            snap = os.path.join(model_cache_dir, "snapshots", rev)
            if os.path.isdir(snap):
                return snap
        # fallback: snapshots 下只有一个时直接用它
        snaps_dir = os.path.join(model_cache_dir, "snapshots")
        if os.path.isdir(snaps_dir):
            snaps = [d for d in os.listdir(snaps_dir) if os.path.isdir(os.path.join(snaps_dir, d))]
            if len(snaps) == 1:
                return os.path.join(snaps_dir, snaps[0])
    except Exception:
        return ""
    return ""


def get_controlnet_inpaint_pipe_sd15(
    *,
    device,
    controlnet_cond: str,
    controlnet_strength: float,
    ip_adapter_strength: float,
    ip_adapter_path: str,
    num_tokens: int,
    ip_adapter_weights: dict,
    torch_dtype=torch.float16,
):
    """
    diffusers: StableDiffusionControlNetInpaintPipeline (SD1.5) + ControlNet + IP-Adapter attention processors。

    说明：
    - base inpaint: runwayml/stable-diffusion-inpainting（SD1.5）
    - controlnet: lllyasviel/control_v11p_sd15_canny 或 lllyasviel/control_v11f1p_sd15_depth
    - IP-Adapter：使用本项目 ip_adapter 下的 setup_ipadapter_attention_processors，把 image embeds 拼到 prompt embeds 后面

    离线支持：
    - 可通过环境变量指定本地模型目录（diffusers 格式）：
      - SD15_INPAINT_MODEL_DIR
      - SD15_CONTROLNET_CANNY_DIR
      - SD15_CONTROLNET_DEPTH_DIR
    """
    # 默认使用你截图里的 HF cache（/data/23009960/hub/...）；若设置了 env，则 env 优先生效。
    hub_root = os.environ.get("HF_HUB_CACHE_ROOT", "/data/23009960/hub")
    default_base = _hf_cache_snapshot_dir(os.path.join(hub_root, "models--runwayml--stable-diffusion-inpainting"))
    base_local = os.environ.get("SD15_INPAINT_MODEL_DIR") or default_base
    base_id, base_local_only = _pick_local_or_hf_id(base_local, "runwayml/stable-diffusion-inpainting")

    if controlnet_cond == "depth":
        default_cn = _hf_cache_snapshot_dir(os.path.join(hub_root, "models--lllyasviel--control_v11f1p_sd15_depth"))
        cn_local = os.environ.get("SD15_CONTROLNET_DEPTH_DIR") or default_cn
        cn_id, cn_local_only = _pick_local_or_hf_id(cn_local, "lllyasviel/control_v11f1p_sd15_depth")
    else:
        default_cn = _hf_cache_snapshot_dir(os.path.join(hub_root, "models--lllyasviel--control_v11p_sd15_canny"))
        cn_local = os.environ.get("SD15_CONTROLNET_CANNY_DIR") or default_cn
        cn_id, cn_local_only = _pick_local_or_hf_id(cn_local, "lllyasviel/control_v11p_sd15_canny")

    try:
        controlnet = ControlNetModel.from_pretrained(
            cn_id,
            torch_dtype=torch_dtype,
            local_files_only=cn_local_only,
        ).to(device)
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            base_id,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            safety_checker=None,
            local_files_only=base_local_only,
        ).to(device)
    except Exception as e:
        raise RuntimeError(
            "Failed to load SD1.5 ControlNet Inpaint pipeline. "
            "If you are offline, please download the models in diffusers format and set env vars:\n"
            "  - SD15_INPAINT_MODEL_DIR\n"
            "  - SD15_CONTROLNET_CANNY_DIR (when controlnet_cond=canny)\n"
            "  - SD15_CONTROLNET_DEPTH_DIR (when controlnet_cond=depth)\n"
            f"controlnet_cond={controlnet_cond}"
        ) from e

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    # keep a default conditioning scale on the pipeline (used by apply_controlnet_inpaint fallback)
    pipe._easitex_default_controlnet_conditioning_scale = float(controlnet_strength)

    # memory helpers (best-effort)
    try:
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    except Exception:
        pass

    setup_ipadapter_attention_processors(
        pipe=pipe,
        num_tokens=int(num_tokens),
        device=device,
        ip_adapter_weights=ip_adapter_weights,
        ip_adapter_path=ip_adapter_path,
        dtype=next(pipe.unet.parameters()).dtype,
    )
    for attn_processor in pipe.unet.attn_processors.values():
        if isinstance(attn_processor, (IPAttnProcessor, IPAttnProcessor2_0)):
            attn_processor.scale = float(ip_adapter_strength)

    return pipe


def get_controlnet_inpaint_pipe_sd15_refattn(
    *,
    device,
    controlnet_cond: str,
    controlnet_strength: float,
    ip_adapter_strength: float,
    ip_adapter_path: str,
    num_tokens: int,
    ip_adapter_weights: dict,
    torch_dtype=torch.float16,
):
    """
    SD1.5 ControlNet Inpaint pipeline with Reference Attention enabled.
    This uses the custom RefAttn pipeline class and keeps IP-Adapter processors active.
    """
    hub_root = os.environ.get("HF_HUB_CACHE_ROOT", "/data/23009960/hub")
    default_base = _hf_cache_snapshot_dir(os.path.join(hub_root, "models--runwayml--stable-diffusion-inpainting"))
    base_local = os.environ.get("SD15_INPAINT_MODEL_DIR") or default_base
    base_id, base_local_only = _pick_local_or_hf_id(base_local, "runwayml/stable-diffusion-inpainting")

    if controlnet_cond == "depth":
        default_cn = _hf_cache_snapshot_dir(os.path.join(hub_root, "models--lllyasviel--control_v11f1p_sd15_depth"))
        cn_local = os.environ.get("SD15_CONTROLNET_DEPTH_DIR") or default_cn
        cn_id, cn_local_only = _pick_local_or_hf_id(cn_local, "lllyasviel/control_v11f1p_sd15_depth")
    else:
        default_cn = _hf_cache_snapshot_dir(os.path.join(hub_root, "models--lllyasviel--control_v11p_sd15_canny"))
        cn_local = os.environ.get("SD15_CONTROLNET_CANNY_DIR") or default_cn
        cn_id, cn_local_only = _pick_local_or_hf_id(cn_local, "lllyasviel/control_v11p_sd15_canny")

    try:
        controlnet = ControlNetModel.from_pretrained(
            cn_id,
            torch_dtype=torch_dtype,
            local_files_only=cn_local_only,
        ).to(device)
        pipe = StableDiffusionControlNetInpaintRefAttnPipeline.from_pretrained(
            base_id,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            safety_checker=None,
            local_files_only=base_local_only,
        ).to(device)
    except Exception as e:
        raise RuntimeError(
            "Failed to load SD1.5 ControlNet Inpaint RefAttn pipeline. "
            "If you are offline, please download the models in diffusers format and set env vars:\n"
            "  - SD15_INPAINT_MODEL_DIR\n"
            "  - SD15_CONTROLNET_CANNY_DIR (when controlnet_cond=canny)\n"
            "  - SD15_CONTROLNET_DEPTH_DIR (when controlnet_cond=depth)\n"
            f"controlnet_cond={controlnet_cond}"
        ) from e

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    # keep a default conditioning scale on the pipeline (used by callers)
    pipe._easitex_default_controlnet_conditioning_scale = float(controlnet_strength)

    try:
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    except Exception:
        pass

    setup_ipadapter_attention_processors(
        pipe=pipe,
        num_tokens=int(num_tokens),
        device=device,
        ip_adapter_weights=ip_adapter_weights,
        ip_adapter_path=ip_adapter_path,
        dtype=next(pipe.unet.parameters()).dtype,
    )
    for attn_processor in pipe.unet.attn_processors.values():
        if isinstance(attn_processor, (IPAttnProcessor, IPAttnProcessor2_0)):
            attn_processor.scale = float(ip_adapter_strength)

    return pipe


def get_inpainting(device):
    print("=> initializing Inpainting...")

    # Prefer local diffusers-format checkpoint to avoid requiring HuggingFace Hub access.
    # Repo layout: easi-tex/lib/diffusion_helper.py -> easi-tex/models/stable-diffusion-2-inpainting/
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    local_model_dir = os.environ.get(
        "SD2_INPAINT_MODEL_DIR",
        os.path.join(repo_root, "models", "stable-diffusion-2-inpainting"),
    )

    model_id_or_path = local_model_dir if os.path.isdir(local_model_dir) else "stabilityai/stable-diffusion-2-inpainting"
    local_only = os.path.isdir(local_model_dir)

    try:
        model = StableDiffusionInpaintPipeline.from_pretrained(
                model_id_or_path,
            torch_dtype=torch.float16,
                local_files_only=local_only,
                safety_checker=None,
        ).to(device)
    except Exception as e:
        # Give a clearer hint for offline environments.
        raise RuntimeError(
            "Failed to load SD2 inpainting pipeline. "
            "If you are offline / cannot access HuggingFace Hub, make sure the model exists at "
            f"'{local_model_dir}' (diffusers format), or set env SD2_INPAINT_MODEL_DIR to the correct path."
        ) from e

    return model

def get_text2image(device):
    print("=> initializing Inpainting...")

    model_id = "stabilityai/stable-diffusion-2"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    model = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16).to(device)

    return model


@torch.no_grad()
def apply_controlnet_depth(model, ddim_sampler, 
    init_image, prompt, strength, ddim_steps,
    generate_mask_image, keep_mask_image, depth_map_np, 
    a_prompt, n_prompt, guidance_scale, seed, eta, num_samples,
    device, blend=0, save_memory=False, pos_img_prompt_embeds=None, neg_img_prompt_embeds=None):
    """
        Use Stable Diffusion 2 to generate image

        Arguments:
            args: input arguments
            model: Stable Diffusion 2 model
            init_image_tensor: input image, torch.FloatTensor of shape (1, H, W, 3)
            mask_tensor: depth map of the input image, torch.FloatTensor of shape (1, H, W, 1)
            depth_map_np: depth map of the input image, torch.FloatTensor of shape (1, H, W)
    """

    print("=> generating ControlNet Depth RePaint image...")


    # Stable Diffusion 2 receives PIL.Image
    # NOTE Stable Diffusion 2 returns a PIL.Image object
    # image and mask_image should be PIL images.
    # The mask structure is white for inpainting and black for keeping as is
    diffused_image_np = process(
        model, ddim_sampler,
        np.array(init_image), prompt, a_prompt, n_prompt, num_samples,
        ddim_steps, guidance_scale, seed, eta, 
        strength=strength, detected_map=depth_map_np, unknown_mask=np.array(generate_mask_image), save_memory=save_memory,
        pos_img_prompt_embeds=pos_img_prompt_embeds, neg_img_prompt_embeds=neg_img_prompt_embeds
    )[0]

    init_image = init_image.convert("RGB")
    diffused_image = Image.fromarray(diffused_image_np).convert("RGB")

    if blend > 0 and transforms.ToTensor()(keep_mask_image).sum() > 0:
        print("=> blending the generated region...")
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        keep_image_np = np.array(init_image).astype(np.uint8)
        keep_image_np_dilate = cv2.dilate(keep_image_np, kernel, iterations=1)

        keep_mask_np = np.array(keep_mask_image).astype(np.uint8)
        keep_mask_np_dilate = cv2.dilate(keep_mask_np, kernel, iterations=1)

        generate_image_np = np.array(diffused_image).astype(np.uint8)

        overlap_mask_np = np.array(generate_mask_image).astype(np.uint8)
        overlap_mask_np *= keep_mask_np_dilate
        print("=> blending {} pixels...".format(np.sum(overlap_mask_np)))

        overlap_keep = keep_image_np_dilate[overlap_mask_np == 1]
        overlap_generate = generate_image_np[overlap_mask_np == 1]

        overlap_np = overlap_keep * blend + overlap_generate * (1 - blend)

        generate_image_np[overlap_mask_np == 1] = overlap_np

        diffused_image = Image.fromarray(generate_image_np.astype(np.uint8)).convert("RGB")

    init_image_masked = init_image
    diffused_image_masked = diffused_image

    return diffused_image, init_image_masked, diffused_image_masked


@torch.no_grad()
def process_inpaint_withref(
    pipe: StableDiffusionControlNetInpaintRefAttnPipeline,
    input_image: np.ndarray,
    prompt: str,
    a_prompt: str,
    n_prompt: str,
    num_samples: int,
    ddim_steps: int,
    scale: float,
    seed: int,
    eta: float,
    *,
    strength: float = 1.0,
    detected_map: np.ndarray | None = None,
    unknown_mask: np.ndarray | None = None,
    reference_image: Image.Image | None = None,
    image_prompt_embeds: torch.Tensor | None = None,
    negative_image_prompt_embeds: torch.Tensor | None = None,
    controlnet_conditioning_scale: float = 1.0,
    depth_pad: int = 10,
):
    """
    Diffusers inpaint process with reference attention.
    Mask handling mirrors ControlNet/gradio_depth2image.py (unknown + background, dilated).
    """
    if num_samples != 1:
        raise ValueError("process_inpaint_withref currently supports num_samples == 1 only.")
    if input_image.ndim != 3 or input_image.shape[-1] != 3:
        raise ValueError(f"process_inpaint_withref expects input_image shape (H,W,3), got {input_image.shape}.")

    H, W, _ = input_image.shape
    if unknown_mask is None:
        raise ValueError("process_inpaint_withref requires unknown_mask (H,W) in {0,255}.")

    pos_prompt = prompt + ", " + a_prompt if (a_prompt is not None and len(a_prompt) > 0) else prompt
    neg_prompt = n_prompt if n_prompt is not None else ""

    input_image_rgb = np.array(Image.fromarray(input_image).convert("RGB"))
    # control map
    if detected_map is None:
        detected_map_resized = np.zeros((H, W, 3), dtype=np.uint8)
    else:
        dm = np.clip(np.asarray(detected_map), 0, 255).astype(np.uint8)
        if dm.ndim == 2:
            dm = np.repeat(dm[..., None], 3, axis=2)
        if dm.shape[-1] == 1:
            dm = np.repeat(dm, 3, axis=2)
        if dm.shape[-1] > 3:
            dm = dm[..., :3]
        detected_map_resized = cv2.resize(dm, (W, H), interpolation=cv2.INTER_LINEAR)

    if unknown_mask is not None:
        # unknown + background mask (mirror gradio_depth2image)
        unknown_mask_np = np.asarray(unknown_mask).astype(np.uint8)
        if unknown_mask_np.shape != (H, W):
            raise ValueError(f"unknown_mask shape must be (H,W)={(H,W)}, got {unknown_mask_np.shape}.")

        detected_map_image = Image.fromarray(detected_map_resized).convert("L")
        detected_map_np = np.array(detected_map_image)
        # Only merge "background" when the control map actually uses `depth_pad`
        # as its background marker (depth mode). For canny mode this marker is
        # typically absent, so we skip this merge to avoid unintended mask drift.
        if np.any(detected_map_np == depth_pad):
            background_mask = (detected_map_np == depth_pad).astype(np.float32) * 255  # 0-255
        else:
            background_mask = np.zeros_like(unknown_mask_np, dtype=np.float32)
        unknown_mask_image = np.clip(
            unknown_mask_np.astype(np.float32) + background_mask, 0, 255
        ).astype(np.uint8)

    # dilate mask for inpaint
    unknown_mask_dilate = cv2.dilate(
        unknown_mask_image, kernel=np.ones((5, 5), np.uint8), iterations=2
    )
    m_dilate = (unknown_mask_dilate > 0).astype(np.float32)  # 1=unknown, 0=known
    m = (unknown_mask_image > 0).astype(np.float32)  # non-dilated for masked_image
    masked_image_np = (input_image_rgb.astype(np.float32) * (1.0 - m[..., None])).clip(0, 255).astype(np.uint8)
    
    masked_image_pil = Image.fromarray(masked_image_np).convert("RGB")
    image_pil = Image.fromarray(input_image_rgb).convert("RGB")
    mask_pil = Image.fromarray((m_dilate * 255.0).astype(np.uint8)).convert("L")
    control_pil = Image.fromarray(detected_map_resized).convert("RGB")
    ref_pil = reference_image.convert("RGB") if reference_image is not None else image_pil

    gen = torch.Generator(device=next(pipe.unet.parameters()).device).manual_seed(int(seed))

    call_kwargs = dict(
        prompt=pos_prompt,
        negative_prompt=neg_prompt,
        image=image_pil,
        mask_image=mask_pil,
        control_image=control_pil,
        reference_image=ref_pil,
        num_inference_steps=int(ddim_steps),
        guidance_scale=float(scale),
        strength=float(strength),
        eta=float(eta),
        generator=gen,
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        output_type="pil",
        cross_attention_kwargs={},
    )
    if image_prompt_embeds is not None and negative_image_prompt_embeds is not None:
        call_kwargs["image_prompt_embeds"] = image_prompt_embeds
        call_kwargs["negative_image_prompt_embeds"] = negative_image_prompt_embeds
    if masked_image_pil is not None:
        call_kwargs["masked_image_pil"] = masked_image_pil
    out = pipe(**call_kwargs).images[0]
    return [np.array(out.convert("RGB")).astype(np.uint8)]


@torch.no_grad()
def apply_inpaintcontrolnetrefattn_depth(
    model: StableDiffusionControlNetInpaintRefAttnPipeline,
    init_image,
    prompt,
    strength,
    ddim_steps,
    generate_mask_image,
    keep_mask_image,
    depth_map_np,
    a_prompt,
    n_prompt,
    guidance_scale,
    seed,
    eta,
    num_samples,
    device,
    blend=0,
    save_memory=False,
    pos_img_prompt_embeds=None,
    neg_img_prompt_embeds=None,
    *,
    reference_image: Image.Image = None,
    controlnet_conditioning_scale: float = None,
):
    """
    Ref-Attn + ControlNet inpaint wrapper (depth/canny), using process_inpaint_withref.
    """
    print("=> generating ControlNet Inpaint (RefAttn) image...")

    base_image = init_image.convert("RGB")
    if reference_image is None:
        reference_image = base_image

    if controlnet_conditioning_scale is None:
        controlnet_conditioning_scale = float(
            getattr(model, "_easitex_default_controlnet_conditioning_scale", 1.0)
        )

    diffused_image_np = process_inpaint_withref(
        model,
        np.array(base_image),
        prompt,
        a_prompt,
        n_prompt,
        num_samples,
        ddim_steps,
        guidance_scale,
        seed,
        eta,
        strength=strength,
        detected_map=depth_map_np,
        unknown_mask=np.array(generate_mask_image),
        reference_image=reference_image,
        image_prompt_embeds=pos_img_prompt_embeds,
        negative_image_prompt_embeds=neg_img_prompt_embeds,
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
    )[0]

    # Keep non-generate area exactly the same as the input render.
    # Without this compose step, inpaint pipelines can still shift tones globally,
    # which often makes the background look darker even when generate_mask is small.
    gen_mask = (np.array(generate_mask_image.convert("L")).astype(np.float32) / 255.0)[..., None]
    base_np = np.array(base_image).astype(np.float32)
    diff_np = diffused_image_np.astype(np.float32)
    diffused_image_np = (gen_mask * diff_np + (1.0 - gen_mask) * base_np).clip(0, 255).astype(np.uint8)
    diffused_image = Image.fromarray(diffused_image_np).convert("RGB")

    if blend > 0 and transforms.ToTensor()(keep_mask_image).sum() > 0:
        print("=> blending the generated region...")
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        keep_image_np = np.array(base_image).astype(np.uint8)
        keep_image_np_dilate = cv2.dilate(keep_image_np, kernel, iterations=1)

        keep_mask_np = np.array(keep_mask_image).astype(np.uint8)
        keep_mask_np_dilate = cv2.dilate(keep_mask_np, kernel, iterations=1)

        generate_image_np = np.array(diffused_image).astype(np.uint8)

        overlap_mask_np = np.array(generate_mask_image).astype(np.uint8)
        overlap_mask_np *= keep_mask_np_dilate
        print("=> blending {} pixels...".format(np.sum(overlap_mask_np)))

        overlap_keep = keep_image_np_dilate[overlap_mask_np == 1]
        overlap_generate = generate_image_np[overlap_mask_np == 1]

        overlap_np = overlap_keep * blend + overlap_generate * (1 - blend)

        generate_image_np[overlap_mask_np == 1] = overlap_np

        diffused_image = Image.fromarray(generate_image_np.astype(np.uint8)).convert("RGB")

    init_image_masked = init_image
    diffused_image_masked = diffused_image
    return diffused_image, init_image_masked, diffused_image_masked
@torch.no_grad()
def process_inpaint(
    pipe: StableDiffusionControlNetInpaintPipeline,
    input_image: np.ndarray,
    prompt: str,
    a_prompt: str,
    n_prompt: str,
    num_samples: int,
    ddim_steps: int,
    scale: float,
    seed: int,
    eta: float,
    *,
    strength: float = 1.0,
    detected_map: np.ndarray | None = None,
    unknown_mask: np.ndarray | None = None,
    save_memory: bool = False,
    pos_img_prompt_embeds: torch.Tensor | None = None,
    neg_img_prompt_embeds: torch.Tensor | None = None,
    controlnet_conditioning_scale: float = 1.0,
    depth_pad: int = 10,
    merge_background: bool = True,
    dilate_mask: bool = True,
    debug_dir: str | None = None,
):
    """
    Diffusers inpaint process (SD1.5): StableDiffusionControlNetInpaintPipeline + (optional) ControlNet condition.

    This mirrors the high-level structure of `models/ControlNet/gradio_depth2image.py:process`,
    but uses the diffusers *inpaint* pipeline (UNet has inpaint channels) instead of ControlLDM.

    Key: build a "masked_image" so the model explicitly sees the known texture context.
      - unknown_mask values must be {0,255}, where 255 means "to inpaint/fill".
    """

    if num_samples != 1:
        raise ValueError("process_inpaint currently supports num_samples == 1 only.")
    if unknown_mask is None:
        raise ValueError("process_inpaint requires unknown_mask (H,W) in {0,255}.")
    if input_image.ndim != 3 or input_image.shape[-1] != 3:
        raise ValueError(f"process_inpaint expects input_image shape (H,W,3), got {input_image.shape}.")

    H, W, _ = input_image.shape
    
    # Process input_image (mirror process() lines 118-124)
    # Ensure RGB format and proper array conversion
    input_image_rgb = np.array(Image.fromarray(input_image).convert("RGB"))
    
    # Process unknown_mask (mirror process() lines 182-223)
    unknown_mask_np = np.asarray(unknown_mask).astype(np.uint8)
    if unknown_mask_np.shape != (H, W):
        raise ValueError(f"unknown_mask shape must be (H,W)={(H,W)}, got {unknown_mask_np.shape}.")

    # Step 4: Process detected_map (control condition) - mirror process() lines 138-147
    # This must be done BEFORE unknown_mask processing to compute background_mask
    if detected_map is None:
        detected_map_resized = np.zeros((H, W, 3), dtype=np.uint8)
        detected_map_np = None
    else:
        dm = np.clip(np.asarray(detected_map), 0, 255).astype(np.uint8)
        if dm.ndim == 2:
            dm = np.repeat(dm[..., None], 3, axis=2)
        if dm.shape[-1] == 1:
            dm = np.repeat(dm, 3, axis=2)
        if dm.shape[-1] > 3:
            dm = dm[..., :3]
        detected_map_resized = cv2.resize(dm, (W, H), interpolation=cv2.INTER_LINEAR)
        # Keep single-channel version for background_mask computation (mirror process() line 191-192)
        detected_map_image = Image.fromarray(detected_map_resized).convert("L")
        detected_map_np = np.array(detected_map_image)
    
    # Process unknown_mask with optional background_mask merge (mirror process() lines 182-202)
    # target: unknown region + background (HACK: basically generate everything except known region)
    if detected_map_np is not None and merge_background:
        background_mask = (detected_map_np == depth_pad).astype(np.float32) * 255  # 0-255
        unknown_mask_image = np.clip(
            unknown_mask_np.astype(np.float32) + background_mask, 0, 255
        ).astype(np.uint8)
    else:
        # No detected_map: use unknown_mask as-is
        unknown_mask_image = unknown_mask_np

    # Dilate unknown_mask_image (mirror process() lines 221-223)
    if dilate_mask:
        unknown_mask_for_pipe = cv2.dilate(
            unknown_mask_image, kernel=np.ones((5, 5), np.uint8), iterations=2
        )
    else:
        unknown_mask_for_pipe = unknown_mask_image
    
    # Step 1: build masked_image (unknown region removed) - use dilated mask for consistency
    m_dilate = (unknown_mask_for_pipe > 0).astype(np.float32)  # 1=unknown, 0=known
    m = (unknown_mask_image > 0).astype(np.float32)  # non-dilated for masked_image
    masked_image_np = (input_image_rgb.astype(np.float32) * (1.0 - m[..., None])).clip(0, 255).astype(np.uint8)

    # NOTE (important):
    # diffusers inpaint pipelines generally expect `image` to be the ORIGINAL image.
    # The pipeline will compute its own masked_image = image * (1 - mask).
    # We still construct `masked_image_np` and, if supported by the installed diffusers version,
    # pass it explicitly via `masked_image=...` for determinism / consistency with "texture inpaint".
    image_pil = Image.fromarray(input_image_rgb).convert("RGB")
    masked_image_pil = Image.fromarray(masked_image_np).convert("RGB")
    
    # Use dilated mask for pipeline input (mirror process() behavior)
    mask_pil = Image.fromarray((m_dilate * 255.0).astype(np.uint8)).convert("L")

    # control_image for ControlNet (already processed above)
    control_pil = Image.fromarray(detected_map_resized).convert("RGB")
    
    # Debug: save masked_image_pil, mask_pil, and control_pil if debug_dir is provided
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        import time
        timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
        masked_image_pil.save(os.path.join(debug_dir, f"debug_{timestamp}_masked_image.png"))
        print(f"=> [DEBUG] Saved masked_image to: {debug_dir}/debug_{timestamp}_masked_image.png")
        mask_pil.save(os.path.join(debug_dir, f"debug_{timestamp}_mask_pil.png"))
        print(f"=> [DEBUG] Saved mask_pil to: {debug_dir}/debug_{timestamp}_mask_pil.png")
        control_pil.save(os.path.join(debug_dir, f"debug_{timestamp}_control_pil.png"))
        print(f"=> [DEBUG] Saved control_pil to: {debug_dir}/debug_{timestamp}_control_pil.png")

    # Text condition + IP-Adapter image prompt embeds
    pos_prompt = prompt + ", " + a_prompt if (a_prompt is not None and len(a_prompt) > 0) else prompt
    neg_prompt = n_prompt if n_prompt is not None else ""

    exec_device = next(pipe.unet.parameters()).device
    gen = torch.Generator(device=exec_device)
    if int(seed) == -1:
        # match "random seed" behavior without importing random
        seed = int(torch.randint(0, 65536, (1,), device=exec_device).item())
    gen.manual_seed(int(seed))

    # Encode prompt -> embeds (then concat IP-Adapter embeds if provided)
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=pos_prompt,
        negative_prompt=neg_prompt,
        device=exec_device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )

    if (pos_img_prompt_embeds is not None) and (neg_img_prompt_embeds is not None):
        to_dtype = prompt_embeds.dtype
        bsz = prompt_embeds.shape[0]
        ip_pos = pos_img_prompt_embeds.to(to_dtype)
        ip_neg = neg_img_prompt_embeds.to(to_dtype)
        if ip_pos.shape[0] != bsz:
            ip_pos = ip_pos.repeat(bsz, 1, 1)
        if ip_neg.shape[0] != negative_prompt_embeds.shape[0]:
            ip_neg = ip_neg.repeat(negative_prompt_embeds.shape[0], 1, 1)
        prompt_embeds = torch.cat([prompt_embeds, ip_pos], dim=1)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds, ip_neg], dim=1)

    # Step 2/3/5/6:
    # Diffusers' inpaint pipeline internally:
    # - encodes (image, masked_image) to latents (x0, x0_masked)
    # - builds mask_latents (1ch) and concatenates them for the inpaint UNet
    # so we pass (image_pil, mask_pil) and let the pipeline do "true inpaint" with the correct channel layout.
    call_kwargs = dict(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        image=image_pil,
        mask_image=mask_pil,
        control_image=control_pil,
        num_inference_steps=int(ddim_steps),
        guidance_scale=float(scale),
        strength=float(strength),
        generator=gen,
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        output_type="pil",
    )

    # best-effort: only schedulers that support eta will use it
    if eta is not None:
        call_kwargs["eta"] = float(eta)

    # best-effort: pass explicit masked_image if supported by this diffusers version
    try:
        out = pipe(**call_kwargs, masked_image=masked_image_pil).images[0]
    except TypeError:
        out = pipe(**call_kwargs).images[0]
    return [np.array(out.convert("RGB")).astype(np.uint8)]


@torch.no_grad()
def apply_controlnet_inpaint_withReference(
    model,
    init_image,
    prompt,
    strength,
    ddim_steps,
    generate_mask_image,
    keep_mask_image,
    a_prompt,
    n_prompt,
    guidance_scale,
    seed,
    eta,
    num_samples,
    device,
    blend=0,
    save_memory=False,
    pos_img_prompt_embeds=None,
    neg_img_prompt_embeds=None,
    *,
    reference_image: Image.Image = None,
    control_image_np=None,
    controlnet_conditioning_scale: float = None,
    strict_compose: bool = True,
):
    """
    diffusers: SD1.5 StableDiffusionControlNetInpaintPipeline 版本的“只在 unknown/new mask 上生成”。

    设计目标（保持与你要求一致）：
    - 输入 init_image 是“从当前纹理渲染出来的视角图”，它包含 old mask 区域的已有纹理；
    - generate_mask_image 指定本视角还缺失/需要生成的区域（白=生成，黑=保留）；
    - 生成后做一次严格合成：mask 外区域完全保持 init_image（保证“根据 old mask 上的纹理”）。

    额外：
    - ControlNet 条件由 control_image_np 提供（depth 或 edges，HWC / 0-255）；
    - IP-Adapter image embeds 通过拼接到 prompt embeds 序列维度注入（pos_img_prompt_embeds / neg_img_prompt_embeds）。
    """
    if num_samples != 1:
        # 当前项目所有调用点都传 1；为了行为可控，这里只支持 1
        raise ValueError("apply_controlnet_inpaint_withReference currently supports num_samples == 1 only.")

    print("=> generating diffusers SD1.5 ControlNet Inpainting image (with reference attention)...")

    pos_prompt = prompt + ", " + a_prompt if (a_prompt is not None and len(a_prompt) > 0) else prompt
    neg_prompt = n_prompt if n_prompt is not None else ""

    base_image = init_image.convert("RGB")
    out_w, out_h = base_image.size

    # mask image (white=inpaint, black=keep)
    mask_image = generate_mask_image.convert("L").resize((out_w, out_h), Image.Resampling.NEAREST)

    # encode prompt (prefer prompt_embeds path, but fall back to prompt strings if needed)
    prompt_embeds = None
    negative_prompt_embeds = None
    if hasattr(model, "encode_prompt"):
        try:
            prompt_embeds, negative_prompt_embeds = model.encode_prompt(
                prompt=pos_prompt,
                negative_prompt=neg_prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
            )
        except TypeError:
            prompt_embeds, negative_prompt_embeds = None, None

    # If using IP-Adapter processors, we require encoder_hidden_states to already
    # include image tokens. Auto-padding is disabled; raise to avoid silent truncation.
    try:
        ip_procs = [
            p
            for p in getattr(model.unet, "attn_processors", {}).values()
            if isinstance(p, (IPAttnProcessor, IPAttnProcessor2_0))
        ]
    except Exception:
        ip_procs = []

    if ip_procs:
        num_tokens = max(getattr(p, "num_tokens", 0) for p in ip_procs)
        if num_tokens > 0:
            if prompt_embeds is None or getattr(prompt_embeds, "ndim", 0) != 3:
                raise RuntimeError(
                    "IP-Adapter processors are enabled, but prompt_embeds are missing. "
                    "Disable IP-Adapter processors or provide prompt_embeds that already "
                    "include image tokens."
                )
            if int(prompt_embeds.shape[1]) <= int(num_tokens):
                raise RuntimeError(
                    "IP-Adapter processors are enabled, but prompt_embeds do not include "
                    "image tokens. Provide prompt_embeds with appended image tokens or "
                    "disable IP-Adapter processors."
                )

    gen = torch.Generator(device=device).manual_seed(int(seed))

    # run at the render resolution (like apply_controlnet_depth), but ensure sizes are multiples of 8
    # because SD VAE/UNet latents are downsampled by 8.
    run_w = out_w - (out_w % 8)
    run_h = out_h - (out_h % 8)
    if run_w <= 0 or run_h <= 0:
        raise ValueError(f"Invalid image size for SD inpaint: {(out_w, out_h)}")
    base_image_run = base_image if (run_w == out_w and run_h == out_h) else base_image.resize((run_w, run_h), Image.Resampling.LANCZOS)
    mask_run = mask_image if (run_w == out_w and run_h == out_h) else mask_image.resize((run_w, run_h), Image.Resampling.NEAREST)
    # Optional ControlNet control image (only for StableDiffusionControlNetInpaintPipeline)
    control_run = None
    if control_image_np is not None:
        # mimic apply_controlnet_depth preprocessing as much as possible:
        # - keep control image in uint8 space
        # - resize with cv2.INTER_LINEAR (NOT NEAREST) to avoid overly "hard" edges
        ctrl = np.clip(np.array(control_image_np), 0, 255).astype(np.uint8)
        if ctrl.ndim == 2:
            ctrl = np.repeat(ctrl[..., None], 3, axis=2)
        if ctrl.shape[-1] == 1:
            ctrl = np.repeat(ctrl, 3, axis=2)
        if ctrl.shape[0] != out_h or ctrl.shape[1] != out_w:
            ctrl = cv2.resize(ctrl, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        control_image = Image.fromarray(ctrl).convert("RGB")
        # resize to run size
        if run_w == out_w and run_h == out_h:
            control_run = control_image
        else:
            ctrl_run = cv2.resize(
                np.array(control_image).astype(np.uint8),
                (run_w, run_h),
                interpolation=cv2.INTER_LINEAR,
            )
            control_run = Image.fromarray(ctrl_run).convert("RGB")

    # ---------------- Reference Attention (replace IP-Adapter) ----------------
    # If no reference_image is provided, default to init_image (so output stays consistent with the current view).
    if reference_image is None:
        reference_image = base_image

    # Enable reference attention on the pipeline UNet (best-effort; no-op if unsupported)
    try:
        ensure_ref_unet_wrapped(model)
    except Exception as _e:
        # If reference attention cannot be enabled, we silently fall back to normal generation.
        pass

    # If UNet is wrapped, set the reference latent for the denoising loop.
    if hasattr(model, "unet") and hasattr(model.unet, "set_cond_lat"):
        # prepare ref image tensor in [-1, 1] and encode to VAE latent space
        ref_img = reference_image.convert("RGB").resize((run_w, run_h), Image.Resampling.LANCZOS)
        ref_np = np.array(ref_img).astype(np.float32) / 127.5 - 1.0  # [-1,1]
        ref_tensor = torch.from_numpy(ref_np).permute(2, 0, 1).unsqueeze(0).to(device)
        ref_tensor = ref_tensor.to(dtype=next(model.vae.parameters()).dtype)
        with torch.inference_mode():
            vae_scale = float(getattr(getattr(model.vae, "config", None), "scaling_factor", 0.18215))

            # IMPORTANT: use deterministic VAE encoding (mode/mean), NOT sampling.
            # Sampling here will inject random noise into reference latent and can destroy reference consistency.
            ref_lat = model.vae.encode(ref_tensor).latent_dist.mode()
            ref_lat = ref_lat * vae_scale

            # unconditional reference latent: encode a zero tensor (same as MeshGen's torch.zeros_like(image))
            zero_tensor = torch.zeros_like(ref_tensor)
            neg_lat = model.vae.encode(zero_tensor).latent_dist.mode()
            neg_lat = neg_lat * vae_scale

            is_cfg = float(guidance_scale) > 1.0
            cond_lat = torch.cat([neg_lat, ref_lat], dim=0) if is_cfg else ref_lat
            model.unet.set_cond_lat(cond_lat, is_cfg_guidance=is_cfg)

            # For inpainting UNet (in_channels=9), the UNet input is:
            #   latent(4) + mask(1) + masked_latent(4)
            # Our reference-attention write-pass needs the same channel structure; otherwise conv_in
            # will error (expected 9, got 4). Some diffusers versions may not allow inferring this
            # reliably from `sample`, so we compute it here and pass it to the wrapper.
            if hasattr(model.unet, "set_inpaint_extra"):
                try:
                    # base image tensor in [-1, 1]
                    base_np = np.array(base_image_run.convert("RGB")).astype(np.float32) / 127.5 - 1.0
                    base_tensor = torch.from_numpy(base_np).permute(2, 0, 1).unsqueeze(0).to(device)
                    base_tensor = base_tensor.to(dtype=ref_tensor.dtype)

                    # mask tensor in [0, 1], where 1 = inpaint region, 0 = keep region
                    m_np = np.array(mask_run.convert("L")).astype(np.float32) / 255.0
                    mask_tensor = torch.from_numpy(m_np).unsqueeze(0).unsqueeze(0).to(device, dtype=base_tensor.dtype)

                    # masked image keeps known region, zeros unknown region
                    masked_img_tensor = base_tensor * (1.0 - mask_tensor)
                    masked_lat = model.vae.encode(masked_img_tensor).latent_dist.mode()
                    masked_lat = masked_lat * vae_scale

                    mask_lat = F.interpolate(mask_tensor, size=masked_lat.shape[-2:], mode="nearest")
                    extra = torch.cat([mask_lat, masked_lat], dim=1).to(dtype=cond_lat.dtype)

                    # match batch with cond_lat (CFG -> 2*B)
                    if extra.shape[0] != cond_lat.shape[0]:
                        if extra.shape[0] == 1:
                            extra = extra.repeat(cond_lat.shape[0], 1, 1, 1)
                        else:
                            extra = extra[: cond_lat.shape[0]]

                    model.unet.set_inpaint_extra(extra)
                except Exception:
                    # fall back to runtime inference inside wrapper (or zero-padding)
                    model.unet.set_inpaint_extra(None)

    # ---------------- Run pipeline ----------------
    if controlnet_conditioning_scale is None:
        controlnet_conditioning_scale = float(getattr(model, "_easitex_default_controlnet_conditioning_scale", 1.0))

    # Prefer prompt_embeds path if supported; else fall back to prompt strings.
    try:
        call_kwargs = dict(
            image=base_image_run,
            mask_image=mask_run,
            num_inference_steps=int(ddim_steps),
            guidance_scale=float(guidance_scale),
            strength=float(strength),
            generator=gen,
            output_type="pil",
        )
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            call_kwargs["prompt_embeds"] = prompt_embeds
            call_kwargs["negative_prompt_embeds"] = negative_prompt_embeds
        else:
            call_kwargs["prompt"] = pos_prompt
            call_kwargs["negative_prompt"] = neg_prompt

        # ControlNet-only args (only if control_run is provided and the pipeline supports it)
        if control_run is not None:
            call_kwargs["control_image"] = control_run
            call_kwargs["controlnet_conditioning_scale"] = float(controlnet_conditioning_scale)

        out = model(**call_kwargs).images[0].resize((out_w, out_h), Image.Resampling.LANCZOS)
    except TypeError:
        # Older pipeline signature (no prompt_embeds, no control_image kwargs)
        out = model(
            prompt=pos_prompt,
            negative_prompt=neg_prompt,
            image=base_image_run,
            mask_image=mask_run,
            height=run_h,
            width=run_w,
            num_inference_steps=int(ddim_steps),
            guidance_scale=float(guidance_scale),
            strength=float(strength),
            generator=gen,
            eta=float(eta),
            output_type="pil",
        ).images[0].resize((out_w, out_h), Image.Resampling.LANCZOS)

    if strict_compose:
        # 严格 inpaint：mask=0 的已知区域完全保持 base_image
        base_np = np.array(base_image).astype(np.float32)
        gen_np = np.array(out.convert("RGB")).astype(np.float32)
        m = (np.array(mask_image).astype(np.float32) / 255.0)[..., None]
        out = Image.fromarray((gen_np * m + base_np * (1.0 - m)).clip(0, 255).astype(np.uint8)).convert("RGB")

    diffused_image = out

    # optional boundary blending (保持与 apply_controlnet_depth 的接口一致)
    if blend > 0 and transforms.ToTensor()(keep_mask_image).sum() > 0:
        print("=> blending the generated region...")
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        keep_image_np = np.array(init_image).astype(np.uint8)
        keep_image_np_dilate = cv2.dilate(keep_image_np, kernel, iterations=1)

        keep_mask_np = (np.array(keep_mask_image).astype(np.uint8) > 127).astype(np.uint8)
        keep_mask_np_dilate = cv2.dilate(keep_mask_np, kernel, iterations=1)

        generate_image_np = np.array(diffused_image).astype(np.uint8)
        gen_mask_np = (np.array(generate_mask_image).astype(np.uint8) > 127).astype(np.uint8)

        overlap_mask_np = gen_mask_np * keep_mask_np_dilate
        print("=> blending {} pixels...".format(np.sum(overlap_mask_np)))

        overlap_keep = keep_image_np_dilate[overlap_mask_np == 1]
        overlap_generate = generate_image_np[overlap_mask_np == 1]
        overlap_np = overlap_keep * blend + overlap_generate * (1 - blend)
        generate_image_np[overlap_mask_np == 1] = overlap_np
        diffused_image = Image.fromarray(generate_image_np.astype(np.uint8)).convert("RGB")

    init_image_masked = init_image
    diffused_image_masked = diffused_image
    return diffused_image, init_image_masked, diffused_image_masked


@torch.no_grad()
def apply_controlnet_inpaint(
    model,
    init_image,
    prompt,
    strength,
    ddim_steps,
    generate_mask_image,
    keep_mask_image,
    a_prompt,
    n_prompt,
    guidance_scale,
    seed,
    eta,
    num_samples,
    device,
    blend=0,
    save_memory=False,
    pos_img_prompt_embeds=None,
    neg_img_prompt_embeds=None,
    *,
    control_image_np=None,
    controlnet_conditioning_scale: float = None,
    strict_compose: bool = True,
    debug_dir: str | None = None,
    merge_background: bool = True,
    dilate_mask: bool = True,
    exclude_keep_from_generate_mask: bool = False,
):
    """
    diffusers: SD1.5 StableDiffusionControlNetInpaintPipeline 版本的“只在 unknown/new mask 上生成”（IP-Adapter 路径）。

    说明：
    - 这个函数是“原版 IP-Adapter”逻辑：把 image embeds 拼接到 prompt_embeds 的 token 维度。
    - ControlNet 条件由 control_image_np 提供（depth 或 edges，HWC / 0-255）。

    若你想用 MeshGen 风格 reference attention，请调用 apply_controlnet_inpaint_withReference。
    """
    if num_samples != 1:
        raise ValueError("apply_controlnet_inpaint currently supports num_samples == 1 only.")

    print("=> generating diffusers SD1.5 ControlNet INPAINT image (IP-Adapter)...")

    base_image = init_image.convert("RGB")
    out_w, out_h = base_image.size

    # Debug: save input images and masks if debug_dir is provided
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        import time
        timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
        
        # Save input image
        base_image.save(os.path.join(debug_dir, f"debug_{timestamp}_input_image.png"))
        print(f"=> [DEBUG] Saved input image to: {debug_dir}/debug_{timestamp}_input_image.png")
        
        # Save keep mask (existing/known region)
        keep_mask_resized = keep_mask_image.convert("L").resize((out_w, out_h), Image.Resampling.NEAREST)
        keep_mask_resized.save(os.path.join(debug_dir, f"debug_{timestamp}_keep_mask.png"))
        print(f"=> [DEBUG] Saved keep mask to: {debug_dir}/debug_{timestamp}_keep_mask.png")
        
        # Save generate mask (new/unknown region to inpaint)
        generate_mask_resized = generate_mask_image.convert("L").resize((out_w, out_h), Image.Resampling.NEAREST)
        generate_mask_resized.save(os.path.join(debug_dir, f"debug_{timestamp}_generate_mask.png"))
        print(f"=> [DEBUG] Saved generate mask to: {debug_dir}/debug_{timestamp}_generate_mask.png")
        
        # Optionally save control image if provided
        if control_image_np is not None:
            control_pil = Image.fromarray(
                np.clip(control_image_np, 0, 255).astype(np.uint8)
            ).convert("RGB").resize((out_w, out_h), Image.Resampling.LANCZOS)
            control_pil.save(os.path.join(debug_dir, f"debug_{timestamp}_control_image.png"))
            print(f"=> [DEBUG] Saved control image to: {debug_dir}/debug_{timestamp}_control_image.png")

    # unknown/new mask (white=inpaint, black=keep)
    mask_image = generate_mask_image.convert("L").resize((out_w, out_h), Image.Resampling.NEAREST)

    # Critical: make the actual inpaint region exclude known/existing pixels.
    # This is what prevents "existing texture becomes less" after inpaint.
    if exclude_keep_from_generate_mask:
        keep_resized = keep_mask_image.convert("L").resize((out_w, out_h), Image.Resampling.NEAREST)
        gen_bin = (np.array(mask_image).astype(np.uint8) > 127)
        keep_bin = (np.array(keep_resized).astype(np.uint8) > 127)
        eff_bin = gen_bin & (~keep_bin)
        mask_image = Image.fromarray((eff_bin.astype(np.uint8) * 255)).convert("L")

    if controlnet_conditioning_scale is None:
        controlnet_conditioning_scale = float(
            getattr(model, "_easitex_default_controlnet_conditioning_scale", 1.0)
        )

    diffused_image_np = process_inpaint(
        model,
        np.array(base_image),
        prompt,
        a_prompt,
        n_prompt,
        num_samples,
        int(ddim_steps),
        float(guidance_scale),
        int(seed),
        float(eta),
        strength=float(strength),
        detected_map=control_image_np,
        unknown_mask=np.array(mask_image),
        save_memory=save_memory,
        pos_img_prompt_embeds=pos_img_prompt_embeds,
        neg_img_prompt_embeds=neg_img_prompt_embeds,
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        merge_background=bool(merge_background),
        dilate_mask=bool(dilate_mask),
        debug_dir=debug_dir,
    )[0]

    init_image = base_image
    diffused_image = Image.fromarray(diffused_image_np).convert("RGB").resize(
        (out_w, out_h), Image.Resampling.LANCZOS
    )

    # out = diffused_image

    # if strict_compose:
    #     # 严格 inpaint：mask=0 的已知区域完全保持 base_image
    #     base_np = np.array(base_image).astype(np.float32)
    #     gen_np = np.array(out.convert("RGB")).astype(np.float32)
    #     m = (np.array(mask_image).astype(np.float32) / 255.0)[..., None]
    #     out = Image.fromarray(
    #         (gen_np * m + base_np * (1.0 - m)).clip(0, 255).astype(np.uint8)
    #     ).convert("RGB")

    # diffused_image = out

    # optional boundary blending (保持与 apply_controlnet_depth 的接口一致)
    # Keep the original boundary blending semantics (same as apply_controlnet_depth):
    # - Masks are uint8 in {0,255}. Multiplication overflows to 1 for (255*255) in uint8.
    if blend > 0 and transforms.ToTensor()(keep_mask_image).sum() > 0:
        print("=> blending the generated region...")
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        keep_image_np = np.array(init_image).astype(np.uint8)
        keep_image_np_dilate = cv2.dilate(keep_image_np, kernel, iterations=1)

        keep_mask_np = (np.array(keep_mask_image).astype(np.uint8) > 127).astype(np.uint8)
        keep_mask_np_dilate = cv2.dilate(keep_mask_np, kernel, iterations=1)

        generate_image_np = np.array(diffused_image).astype(np.uint8)
        gen_mask_np = (np.array(generate_mask_image).astype(np.uint8) > 127).astype(np.uint8)

        overlap_mask_np = gen_mask_np * keep_mask_np_dilate
        print("=> blending {} pixels...".format(np.sum(overlap_mask_np)))

        overlap_keep = keep_image_np_dilate[overlap_mask_np == 1]
        overlap_generate = generate_image_np[overlap_mask_np == 1]
        overlap_np = overlap_keep * blend + overlap_generate * (1 - blend)
        generate_image_np[overlap_mask_np == 1] = overlap_np
        diffused_image = Image.fromarray(generate_image_np.astype(np.uint8)).convert("RGB")

    init_image_masked = init_image
    diffused_image_masked = diffused_image
    return diffused_image, init_image_masked, diffused_image_masked


# Optional explicit alias (readability)
apply_controlnet_inpaint_ipadapter = apply_controlnet_inpaint


@torch.no_grad()
def apply_inpainting(model, 
    init_image, mask_image_tensor, prompt, height, width, device):
    """
        Use Stable Diffusion 2 to generate image

        Arguments:
            args: input arguments
            model: Stable Diffusion 2 model
            init_image_tensor: input image, torch.FloatTensor of shape (1, H, W, 3)
            mask_tensor: depth map of the input image, torch.FloatTensor of shape (1, H, W, 1)
            depth_map_tensor: depth map of the input image, torch.FloatTensor of shape (1, H, W)
    """

    print("=> generating Inpainting image...")

    mask_image = mask_image_tensor[0].cpu()
    mask_image = mask_image.permute(2, 0, 1)
    mask_image = transforms.ToPILImage()(mask_image).convert("L")

    # NOTE Stable Diffusion 2 returns a PIL.Image object
    # image and mask_image should be PIL images.
    # The mask structure is white for inpainting and black for keeping as is
    diffused_image = model(
        prompt=prompt, 
        image=init_image.resize((512, 512)), 
        mask_image=mask_image.resize((512, 512)), 
        height=512, 
        width=512
    ).images[0].resize((height, width))

    return diffused_image


@torch.no_grad()
def apply_inpainting_postprocess(model, 
    init_image, mask_image_tensor, prompt, height, width, device):
    """
        Use Stable Diffusion 2 to generate image

        Arguments:
            args: input arguments
            model: Stable Diffusion 2 model
            init_image_tensor: input image, torch.FloatTensor of shape (1, H, W, 3)
            mask_tensor: depth map of the input image, torch.FloatTensor of shape (1, H, W, 1)
            depth_map_tensor: depth map of the input image, torch.FloatTensor of shape (1, H, W)
    """

    print("=> generating Inpainting image...")

    mask_image = mask_image_tensor[0].cpu()
    mask_image = mask_image.permute(2, 0, 1)
    mask_image = transforms.ToPILImage()(mask_image).convert("L")

    # NOTE Stable Diffusion 2 returns a PIL.Image object
    # image and mask_image should be PIL images.
    # The mask structure is white for inpainting and black for keeping as is
    diffused_image = model(
        prompt=prompt, 
        image=init_image.resize((512, 512)), 
        mask_image=mask_image.resize((512, 512)), 
        height=512, 
        width=512
    ).images[0].resize((height, width))

    diffused_image_tensor = torch.from_numpy(np.array(diffused_image)).to(device)

    init_images_tensor = torch.from_numpy(np.array(init_image)).to(device)
    
    init_images_tensor = diffused_image_tensor * mask_image_tensor[0] + init_images_tensor * (1 - mask_image_tensor[0])
    init_image = Image.fromarray(init_images_tensor.cpu().numpy().astype(np.uint8)).convert("RGB")

    return init_image
