import inspect
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
    StableDiffusionControlNetImg2ImgPipeline,
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
from lib.reference_attention_new_4Channel import ensure_ref_unet_wrapped, ReferenceOnlyAttnProc

from lib.pipeline_controlnet_inpaint_refattn import (
    StableDiffusionControlNetInpaintRefAttnPipeline,
)


def _ref_attn_runtime_checks_enabled() -> bool:
    return os.environ.get("EASITEX_REF_ATTN_RUNTIME_CHECKS", "0") != "0"


def _ref_attn_log(message: str) -> None:
    if _ref_attn_runtime_checks_enabled():
        print(f"[RefAttnCheck] {message}")


def get_controlnet_depth(**kwargs):
    print("=> initializing 4-channel ControlNet pipeline...")
    pipe = get_controlnet_img2img_pipe_sd15_refattn(
        device=kwargs.get(
            "device",
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        ),
        controlnet_cond=kwargs.get("controlnet_cond", "depth"),
        controlnet_strength=kwargs.get("controlnet_strength", 1.0),
        ip_adapter_strength=kwargs.get("ip_adapter_strength", 1.0),
        ip_adapter_path=kwargs.get("ip_adapter_path"),
        num_tokens=kwargs.get("ip_adapter_n_tokens", kwargs.get("num_tokens", 16)),
        ip_adapter_weights=kwargs.get("ip_adapter_weights"),
    )

    return pipe, None


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


def _set_ipadapter_processor_scale(unet, scale: float):
    for attn_processor in unet.attn_processors.values():
        base_proc = (
            attn_processor.chained_proc
            if isinstance(attn_processor, ReferenceOnlyAttnProc)
            else attn_processor
        )
        if isinstance(base_proc, (IPAttnProcessor, IPAttnProcessor2_0)):
            base_proc.scale = float(scale)




def get_controlnet_img2img_pipe_sd15_refattn(
    *,
    device,
    controlnet_cond: str,
    controlnet_strength: float,
    ip_adapter_strength: float,
    ip_adapter_path: str,
    num_tokens: int,
    ip_adapter_weights: dict | None = None,
    torch_dtype=torch.float16,
):
    """
    4-channel SD1.5 ControlNet Img2Img pipeline with MeshGen-style Reference Attention.

    Offline support:
    - SD15_BASE_MODEL_DIR
    - SD15_CONTROLNET_CANNY_DIR
    - SD15_CONTROLNET_DEPTH_DIR
    """
    hub_root = os.environ.get("HF_HUB_CACHE_ROOT", "/data/23009960/hub")
    default_base = _hf_cache_snapshot_dir(
        os.path.join(hub_root, "models--runwayml--stable-diffusion-v1-5")
    )
    base_local = os.environ.get("SD15_BASE_MODEL_DIR") or default_base
    base_id, base_local_only = _pick_local_or_hf_id(
        base_local, "runwayml/stable-diffusion-v1-5"
    )

    if controlnet_cond == "depth":
        default_cn = _hf_cache_snapshot_dir(
            os.path.join(hub_root, "models--lllyasviel--control_v11f1p_sd15_depth")
        )
        cn_local = os.environ.get("SD15_CONTROLNET_DEPTH_DIR") or default_cn
        cn_id, cn_local_only = _pick_local_or_hf_id(
            cn_local, "lllyasviel/control_v11f1p_sd15_depth"
        )
    else:
        default_cn = _hf_cache_snapshot_dir(
            os.path.join(hub_root, "models--lllyasviel--control_v11p_sd15_canny")
        )
        cn_local = os.environ.get("SD15_CONTROLNET_CANNY_DIR") or default_cn
        cn_id, cn_local_only = _pick_local_or_hf_id(
            cn_local, "lllyasviel/control_v11p_sd15_canny"
        )

    try:
        controlnet = ControlNetModel.from_pretrained(
            cn_id,
            torch_dtype=torch_dtype,
            local_files_only=cn_local_only,
        ).to(device)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            base_id,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            safety_checker=None,
            local_files_only=base_local_only,
        ).to(device)
    except Exception as e:
        raise RuntimeError(
            "Failed to load SD1.5 ControlNet Img2Img RefAttn pipeline. "
            "If you are offline, please download the models in diffusers format and set env vars:\n"
            "  - SD15_BASE_MODEL_DIR\n"
            "  - SD15_CONTROLNET_CANNY_DIR (when controlnet_cond=canny)\n"
            "  - SD15_CONTROLNET_DEPTH_DIR (when controlnet_cond=depth)\n"
            f"controlnet_cond={controlnet_cond}"
        ) from e

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe._easitex_default_controlnet_conditioning_scale = float(controlnet_strength)
    pipe._easitex_generation_backend = "sd15_controlnet_img2img_refattn"
    pipe._easitex_controlnet_cond = str(controlnet_cond)
    pipe._easitex_merge_background = True

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
    _set_ipadapter_processor_scale(pipe.unet, ip_adapter_strength)

    ensure_ref_unet_wrapped(pipe)
    _set_ipadapter_processor_scale(pipe.unet, ip_adapter_strength)

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
def apply_controlnet_depth_RefAttn(model, ddim_sampler, 
    init_image, prompt, strength, ddim_steps,
    generate_mask_image, keep_mask_image, depth_map_np, 
    a_prompt, n_prompt, guidance_scale, seed, eta, num_samples,
    device, blend=0, save_memory=False, pos_img_prompt_embeds=None, neg_img_prompt_embeds=None,
    reference_image: Image.Image = None, controlnet_conditioning_scale: float = None):
    """
        Use Stable Diffusion 2 to generate image

        Arguments:
            args: input arguments
            model: Stable Diffusion 2 model
            init_image_tensor: input image, torch.FloatTensor of shape (1, H, W, 3)
            mask_tensor: depth map of the input image, torch.FloatTensor of shape (1, H, W, 1)
            depth_map_np: depth map of the input image, torch.FloatTensor of shape (1, H, W)
    """
    print("=> applying ControlNet Depth...")
    if getattr(model, "_easitex_generation_backend", "") == "sd15_controlnet_img2img_refattn":
        return apply_controlnet_img2img_refattn_depth(
            model,
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
            blend=blend,
            save_memory=save_memory,
            pos_img_prompt_embeds=pos_img_prompt_embeds,
            neg_img_prompt_embeds=neg_img_prompt_embeds,
            reference_image=reference_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )

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


def _prepare_scheduler_extra_step_kwargs(scheduler, generator, eta):
    extra_step_kwargs = {}
    try:
        step_sig = inspect.signature(scheduler.step)
        step_args = set(step_sig.parameters.keys())
    except (TypeError, ValueError):
        step_args = set()

    if "eta" in step_args and eta is not None:
        extra_step_kwargs["eta"] = float(eta)
    if "generator" in step_args and generator is not None:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs


def _pil_to_minus_one_to_one_tensor(image: Image.Image, *, device, dtype):
    image_np = np.array(image.convert("RGB")).astype(np.float32) / 127.5 - 1.0
    return (
        torch.from_numpy(image_np)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=dtype)
    )


def _encode_vae_latents(vae, image_tensor, *, generator=None, use_mode=False):
    posterior = vae.encode(image_tensor).latent_dist
    if use_mode:
        latents = posterior.mode()
    else:
        try:
            latents = posterior.sample(generator=generator)
        except TypeError:
            latents = posterior.sample()
    return latents * float(getattr(getattr(vae, "config", None), "scaling_factor", 0.18215))


def _prepare_control_map(detected_map: np.ndarray | None, width: int, height: int):
    if detected_map is None:
        return np.zeros((height, width, 3), dtype=np.uint8)

    dm = np.clip(np.asarray(detected_map), 0, 255).astype(np.uint8)
    if dm.ndim == 2:
        dm = np.repeat(dm[..., None], 3, axis=2)
    if dm.shape[-1] == 1:
        dm = np.repeat(dm, 3, axis=2)
    if dm.shape[-1] > 3:
        dm = dm[..., :3]
    return cv2.resize(dm, (width, height), interpolation=cv2.INTER_LINEAR)


def _prepare_generation_region_masks(
    unknown_mask: np.ndarray | None,
    detected_map: np.ndarray | None,
    width: int,
    height: int,
    latent_shape,
    *,
    device,
    dtype,
    depth_pad: int = 10,
    merge_background: bool = True,
):
    if unknown_mask is None:
        unknown_mask_np = np.ones((height, width), dtype=np.uint8) * 255
    else:
        unknown_mask_np = np.asarray(unknown_mask)
        if unknown_mask_np.ndim == 3:
            unknown_mask_np = np.array(Image.fromarray(unknown_mask_np.astype(np.uint8)).convert("L"))
        unknown_mask_np = cv2.resize(
            unknown_mask_np.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        unknown_mask_np = ((unknown_mask_np > 0).astype(np.uint8) * 255)

    if detected_map is None:
        detected_map_np = np.zeros((height, width), dtype=np.uint8)
    else:
        detected_map_np = np.asarray(detected_map)
        if detected_map_np.ndim == 3:
            detected_map_np = np.array(Image.fromarray(detected_map_np.astype(np.uint8)).convert("L"))
        detected_map_np = cv2.resize(
            detected_map_np.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    if merge_background:
        background_mask = (detected_map_np == depth_pad).astype(np.float32) * 255
    else:
        background_mask = np.zeros_like(unknown_mask_np, dtype=np.float32)
    unknown_mask_image = np.clip(
        unknown_mask_np.astype(np.float32) + background_mask, 0, 255
    ).astype(np.uint8)

    unknown_mask_dilate = cv2.dilate(
        unknown_mask_image, kernel=np.ones((5, 5), np.uint8), iterations=2
    )
    latent_mask = torch.from_numpy(
        (unknown_mask_dilate > 0).astype(np.float32)
    ).unsqueeze(0).unsqueeze(0)
    latent_mask = F.interpolate(
        latent_mask,
        size=(latent_shape[-2], latent_shape[-1]),
        mode="nearest",
    ).to(device=device, dtype=dtype)
    latent_mask = latent_mask.repeat(1, int(latent_shape[1]), 1, 1)
    return {
        "unknown_mask": unknown_mask_np,
        "background_mask": background_mask.astype(np.uint8),
        "merged_mask": unknown_mask_image,
        "dilated_mask": unknown_mask_dilate,
        "latent_mask": latent_mask,
    }


@torch.no_grad()
def process_controlnet_img2img_withref(
    pipe: StableDiffusionControlNetImg2ImgPipeline,
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
    merge_background: bool = True,
):
    if num_samples != 1:
        raise ValueError("process_controlnet_img2img_withref currently supports num_samples == 1 only.")
    if input_image.ndim != 3 or input_image.shape[-1] != 3:
        raise ValueError(
            f"process_controlnet_img2img_withref expects input_image shape (H,W,3), got {input_image.shape}."
        )

    ensure_ref_unet_wrapped(pipe)

    base_image = Image.fromarray(input_image).convert("RGB")
    height, width = input_image.shape[:2]
    if reference_image is None:
        reference_image = base_image

    exec_device = next(pipe.unet.parameters()).device
    unet_dtype = next(pipe.unet.parameters()).dtype
    vae_dtype = next(pipe.vae.parameters()).dtype
    control_dtype = next(pipe.controlnet.parameters()).dtype

    generator = torch.Generator(device=exec_device)
    if int(seed) == -1:
        seed = int(torch.randint(0, 65536, (1,), device=exec_device).item())
    generator.manual_seed(int(seed))

    pos_prompt = prompt + ", " + a_prompt if (a_prompt is not None and len(a_prompt) > 0) else prompt
    neg_prompt = n_prompt if n_prompt is not None else ""
    do_classifier_free_guidance = float(scale) > 1.0
    cfg_layout = "uncond_first" if do_classifier_free_guidance else None

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=pos_prompt,
        negative_prompt=neg_prompt,
        device=exec_device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
    )

    if (image_prompt_embeds is None) ^ (negative_image_prompt_embeds is None):
        raise ValueError(
            "image_prompt_embeds and negative_image_prompt_embeds must be provided together."
        )
    if image_prompt_embeds is not None and negative_image_prompt_embeds is not None:
        to_dtype = prompt_embeds.dtype
        ip_pos = image_prompt_embeds.to(device=exec_device, dtype=to_dtype)
        ip_neg = negative_image_prompt_embeds.to(device=exec_device, dtype=to_dtype)
        if ip_pos.shape[0] != prompt_embeds.shape[0]:
            ip_pos = ip_pos.repeat(prompt_embeds.shape[0], 1, 1)
        if ip_neg.shape[0] != negative_prompt_embeds.shape[0]:
            ip_neg = ip_neg.repeat(negative_prompt_embeds.shape[0], 1, 1)
        prompt_embeds = torch.cat([prompt_embeds, ip_pos], dim=1)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds, ip_neg], dim=1)

    encoder_hidden_states = (
        torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        if do_classifier_free_guidance
        else prompt_embeds
    ).to(device=exec_device, dtype=unet_dtype)

    base_tensor = _pil_to_minus_one_to_one_tensor(
        base_image, device=exec_device, dtype=vae_dtype
    )
    init_latents = _encode_vae_latents(
        pipe.vae, base_tensor, generator=generator, use_mode=False
    ).to(dtype=unet_dtype)

    ref_image = reference_image.convert("RGB").resize(
        (width, height), Image.Resampling.LANCZOS
    )
    ref_tensor = _pil_to_minus_one_to_one_tensor(
        ref_image, device=exec_device, dtype=vae_dtype
    )
    ref_lat = _encode_vae_latents(
        pipe.vae, ref_tensor, generator=None, use_mode=True
    ).to(dtype=unet_dtype)
    neg_lat = _encode_vae_latents(
        pipe.vae,
        torch.full_like(ref_tensor, -1.0),
        generator=None,
        use_mode=True,
    ).to(dtype=unet_dtype)
    cond_lat = (
        torch.cat([neg_lat, ref_lat], dim=0)
        if do_classifier_free_guidance
        else ref_lat
    )

    if _ref_attn_runtime_checks_enabled():
        if do_classifier_free_guidance:
            if negative_prompt_embeds.shape[0] != prompt_embeds.shape[0]:
                raise AssertionError(
                    "CFG batch layout is ambiguous: negative and positive prompt batch sizes do not match."
                )
            if encoder_hidden_states.shape[0] != negative_prompt_embeds.shape[0] + prompt_embeds.shape[0]:
                raise AssertionError(
                    "Helper-constructed encoder_hidden_states does not match the expected [uncond, cond] layout."
                )
            if cond_lat.shape[0] != neg_lat.shape[0] + ref_lat.shape[0]:
                raise AssertionError(
                    "Helper-constructed cond_lat does not match the expected [uncond_ref, cond_ref] layout."
                )
            _ref_attn_log(
                "helper confirmed CFG batch layout: "
                "encoder_hidden_states=[uncond, cond], cond_lat=[uncond_ref, cond_ref], "
                f"batch={encoder_hidden_states.shape[0]}"
            )
        else:
            _ref_attn_log("helper running without CFG; no batch splitting is required")

        _ref_attn_log(
            "helper role separation: "
            f"init_latents_shape={tuple(init_latents.shape)} "
            f"ref_lat_shape={tuple(ref_lat.shape)} "
            f"cond_lat_shape={tuple(cond_lat.shape)}"
        )

    detected_map_resized = _prepare_control_map(detected_map, width, height)
    control_tensor = (
        torch.from_numpy(detected_map_resized)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=exec_device, dtype=control_dtype)
        / 255.0
    )

    scheduler = pipe.scheduler
    scheduler.set_timesteps(int(ddim_steps), device=exec_device)
    timesteps = scheduler.timesteps
    strength = float(np.clip(strength, 0.0, 1.0))
    init_timestep = min(max(int(ddim_steps * strength), 1), int(ddim_steps))
    t_start = max(int(ddim_steps) - init_timestep, 0)
    timesteps = timesteps[t_start:]
    latent_timestep = timesteps[:1].repeat(init_latents.shape[0])

    mask_pack = _prepare_generation_region_masks(
        unknown_mask,
        detected_map,
        width,
        height,
        init_latents.shape,
        device=exec_device,
        dtype=init_latents.dtype,
        depth_pad=depth_pad,
        merge_background=merge_background,
    )
    latent_mask = mask_pack["latent_mask"]
    if _ref_attn_runtime_checks_enabled():
        unknown_pixels = int((mask_pack["unknown_mask"] > 0).sum())
        background_pixels = int((mask_pack["background_mask"] > 0).sum())
        merged_pixels = int((mask_pack["merged_mask"] > 0).sum())
        dilated_pixels = int((mask_pack["dilated_mask"] > 0).sum())
        latent_pixels = float(latent_mask[:, :1].sum().item())
        _ref_attn_log(
            "helper mask prep: "
            f"unknown_pixels={unknown_pixels} "
            f"background_pixels={background_pixels} "
            f"merged_pixels={merged_pixels} "
            f"dilated_pixels={dilated_pixels} "
            f"latent_pixels={latent_pixels:.1f}"
        )
    noise = torch.randn(
        init_latents.shape,
        generator=generator,
        device=exec_device,
        dtype=init_latents.dtype,
    )
    try:
        latents = scheduler.add_noise(init_latents, noise, latent_timestep)
    except Exception:
        latents = scheduler.add_noise(
            init_latents, noise, latent_timestep.reshape(-1)
        )
    extra_step_kwargs = _prepare_scheduler_extra_step_kwargs(
        scheduler, generator, eta
    )

    if hasattr(pipe.unet, "clear_reference_conditioning"):
        pipe.unet.clear_reference_conditioning(reset_progress=True)
    if hasattr(pipe.unet, "set_cond_lat"):
        pipe.unet.set_cond_lat(
            cond_lat,
            is_cfg_guidance=do_classifier_free_guidance,
            cfg_layout=cfg_layout,
        )

    try:
        for step_idx, t in enumerate(timesteps):
            t_batch = t.reshape(1) if isinstance(t, torch.Tensor) and t.ndim == 0 else t
            source_latents = scheduler.add_noise(init_latents, noise, t_batch)
            latents = latent_mask * latents + (1.0 - latent_mask) * source_latents

            latent_model_input = (
                torch.cat([latents] * 2)
                if do_classifier_free_guidance
                else latents
            )
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            control_model_input = control_tensor.repeat(
                latent_model_input.shape[0], 1, 1, 1
            )

            down_block_res_samples, mid_block_res_sample = pipe.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states.to(control_dtype),
                controlnet_cond=control_model_input,
                conditioning_scale=float(controlnet_conditioning_scale),
                return_dict=False,
            )

            if _ref_attn_runtime_checks_enabled() and step_idx < 6:
                timestep_value = (
                    float(t.detach().float().reshape(-1)[0].item())
                    if isinstance(t, torch.Tensor)
                    else float(t)
                )
                _ref_attn_log(
                    "helper denoise step: "
                    f"step={step_idx + 1} "
                    f"timestep={timestep_value} "
                    f"latent_model_input_shape={tuple(latent_model_input.shape)} "
                    f"control_down_blocks={len(down_block_res_samples)} "
                    f"cfg_layout={cfg_layout}"
                )

            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                cross_attention_kwargs={
                    "cfg_layout": cfg_layout,
                },
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + float(scale) * (
                    noise_pred_text - noise_pred_uncond
                )

            latents = scheduler.step(
                noise_pred,
                t,
                latents,
                **extra_step_kwargs,
                return_dict=False,
            )[0]
    finally:
        if hasattr(pipe.unet, "clear_reference_conditioning"):
            pipe.unet.clear_reference_conditioning(reset_progress=True)

    vae_scale = float(getattr(getattr(pipe.vae, "config", None), "scaling_factor", 0.18215))
    image = pipe.vae.decode(
        latents / vae_scale,
        return_dict=False,
    )[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = (
        image[0]
        .detach()
        .float()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )
    return [(image * 255.0).round().clip(0, 255).astype(np.uint8)]


@torch.no_grad()
def apply_controlnet_img2img_refattn_depth(
    model: StableDiffusionControlNetImg2ImgPipeline,
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
    4-channel SD1.5 ControlNet Img2Img path with shared reference attention.
    The known region is preserved during denoising by mixing back noised source
    latents at every step, then composed once more on the final RGB output.
    """
    print("=> generating 4-channel ControlNet Img2Img (RefAttn) image...")

    base_image = init_image.convert("RGB")
    out_w, out_h = base_image.size
    run_w = out_w - (out_w % 8)
    run_h = out_h - (out_h % 8)
    if run_w <= 0 or run_h <= 0:
        raise ValueError(f"Invalid image size for SD img2img: {(out_w, out_h)}")

    if controlnet_conditioning_scale is None:
        controlnet_conditioning_scale = float(
            getattr(model, "_easitex_default_controlnet_conditioning_scale", 1.0)
        )
    merge_background = bool(getattr(model, "_easitex_merge_background", False))
    print(f"merge_background: {merge_background}")

    base_image_run = (
        base_image
        if (run_w == out_w and run_h == out_h)
        else base_image.resize((run_w, run_h), Image.Resampling.LANCZOS)
    )
    generate_mask_run = generate_mask_image.convert("L").resize(
        (run_w, run_h), Image.Resampling.NEAREST
    )
    if reference_image is None:
        reference_image = base_image
    reference_image_run = reference_image.convert("RGB").resize(
        (run_w, run_h), Image.Resampling.LANCZOS
    )

    diffused_image_np = process_controlnet_img2img_withref(
        model,
        np.array(base_image_run),
        prompt,
        a_prompt,
        n_prompt,
        num_samples,
        int(ddim_steps),
        float(guidance_scale),
        int(seed),
        float(eta),
        strength=float(strength),
        detected_map=depth_map_np,
        unknown_mask=np.array(generate_mask_run),
        reference_image=reference_image_run,
        image_prompt_embeds=pos_img_prompt_embeds,
        negative_image_prompt_embeds=neg_img_prompt_embeds,
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        merge_background=merge_background,
    )[0]

    diffused_image = Image.fromarray(diffused_image_np).convert("RGB").resize(
        (out_w, out_h), Image.Resampling.LANCZOS
    )

    # IMPORTANT: unlike the old safety-net composition, keep the decoded image
    # as-is so the unknown/background/dilated seam fixes from latent-space
    # generation are not overwritten back to the original render.

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

    init_image_masked = base_image
    diffused_image_masked = diffused_image
    return diffused_image, init_image_masked, diffused_image_masked






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
