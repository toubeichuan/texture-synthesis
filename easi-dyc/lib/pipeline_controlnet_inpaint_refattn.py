import os
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.controlnet.pipeline_controlnet_inpaint import (
    StableDiffusionControlNetInpaintPipeline,
)

# NOTE: easi-tex local import (project root is expected on sys.path)
from lib.reference_attention import ensure_ref_unet_wrapped, debug_refattn

try:
    from diffusers.utils import replace_example_docstring as _replace_example_docstring
except Exception:
    _replace_example_docstring = None


def safe_replace_example_docstring(example: str):
    # diffusers 没有这个装饰器就 no-op
    if _replace_example_docstring is None:
        def deco(fn):
            return fn
        return deco

    # 如果函数 docstring 为空，也 no-op（避免 None.split 崩）
    def deco(fn):
        if fn.__doc__ is None:
            return fn
        return _replace_example_docstring(example)(fn)

    return deco


def _save_debug_inpaint_inputs(
    init_image,
    mask,
    masked_image,
    out_dir="dbg_inpaint",
    prefix="view",
):
    """
    init_image:   (1,3,H,W) float tensor, usually 0..1
    mask:         (1,1,H,W) float tensor, usually 0..1 (diffusers)
    masked_image: (1,3,H,W) float tensor, usually 0..1 after masking
    """
    os.makedirs(out_dir, exist_ok=True)

    # move to cpu float32
    im = init_image.detach().float().cpu()[0].permute(1, 2, 0).numpy()   # HWC
    m = mask.detach().float().cpu()[0, 0].numpy()                        # HW
    mi = masked_image.detach().float().cpu()[0].permute(1, 2, 0).numpy() # HWC

    im = (im + 1.0) / 2.0
    mi = (mi + 1.0) / 2.0

    # clamp to [0,1] for safe visualization
    im = np.clip(im, 0.0, 1.0)
    mi = np.clip(mi, 0.0, 1.0)
    m = np.clip(m, 0.0, 1.0)

    # 1) raw mask
    Image.fromarray((m * 255).astype(np.uint8)).save(
        os.path.join(out_dir, f"{prefix}_mask.png")
    )

    # 2) overlay: mark mask>=0.5 region as red (what your code treats as "hole")
    overlay = im.copy()
    hole = m >= 0.5
    overlay[hole] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    Image.fromarray((overlay * 255).astype(np.uint8)).save(
        os.path.join(out_dir, f"{prefix}_overlay.png")
    )

    # 3) masked_image
    Image.fromarray((mi * 255).astype(np.uint8)).save(
        os.path.join(out_dir, f"{prefix}_masked_image.png")
    )

    # quick stats
    print(
        f"[DBG] {prefix} mask: min={m.min():.4f} max={m.max():.4f} "
        f"mean={m.mean():.4f} hole_ratio(m>=0.5)={hole.mean():.4f}"
    )
    print(
        f"[DBG] {prefix} init_image: min={im.min():.4f} "
        f"max={im.max():.4f} mean={im.mean():.4f}"
    )
    print(
        f"[DBG] {prefix} masked_image: min={mi.min():.4f} "
        f"max={mi.max():.4f} mean={mi.mean():.4f}"
    )


EXAMPLE_DOC_STRING = """
Examples:
    ```py
    >>> pipe = StableDiffusionControlNetInpaintRefAttnPipeline.from_pretrained(...)
    >>> image = pipe(
    ...     prompt="a cat",
    ...     image=init_image,
    ...     mask_image=mask_image,
    ...     control_image=control_image,
    ...     reference_image=reference_image,
    ...     image_prompt_embeds=ip_pos,
    ...     negative_image_prompt_embeds=ip_neg,
    ... ).images[0]
    ```
"""


class StableDiffusionControlNetInpaintRefAttnPipeline(
    StableDiffusionControlNetInpaintPipeline
):
    @torch.no_grad()
    @safe_replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        control_image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        padding_mask_crop: Optional[int] = None,
        strength: float = 1.0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 0.5,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # ref-attention extras
        reference_image: Optional[PipelineImageInput] = None,
        image_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_image_prompt_embeds: Optional[torch.FloatTensor] = None,
        masked_image_pil: Optional[Image.Image] = None,
        **kwargs,
    ):
        if reference_image is not None:
            ensure_ref_unet_wrapped(self)

        if hasattr(self.unet, "clear_reference_conditioning"):
            self.unet.clear_reference_conditioning(reset_progress=True)

        # ---------------- Reference Attention setup ----------------
        if reference_image is not None:
            if os.environ.get("EASITEX_DEBUG_REF_ATTN", "0") == "1":
                debug_refattn(self)

            # Align preprocessing with base pipeline
            if padding_mask_crop is not None:
                height, width = self.image_processor.get_default_height_width(
                    image, height, width
                )
                crops_coords = self.mask_processor.get_crop_region(
                    mask_image,
                    width,
                    height,
                    pad=padding_mask_crop,
                )
                resize_mode = "fill"
            else:
                crops_coords = None
                resize_mode = "default"

            init_image = self.image_processor.preprocess(
                image,
                height=height,
                width=width,
                crops_coords=crops_coords,
                resize_mode=resize_mode,
            )
            init_image = init_image.to(dtype=torch.float32)

            mask = self.mask_processor.preprocess(
                mask_image,
                height=height,
                width=width,
                resize_mode=resize_mode,
                crops_coords=crops_coords,
            )

            masked_image = None
            if masked_image_pil is not None:
                masked_image = self.image_processor.preprocess(
                    masked_image_pil,
                    height=height,
                    width=width,
                    crops_coords=crops_coords,
                    resize_mode=resize_mode,
                )
                masked_image = masked_image.to(dtype=torch.float32)

            _, _, height, width = init_image.shape

            # ===== Step1/2 debug dump (save mask + overlay + masked_image) =====
            if masked_image is not None:
                try:
                    _save_debug_inpaint_inputs(
                        init_image=init_image,
                        mask=mask,
                        masked_image=masked_image,
                        out_dir="dbg_inpaint",
                        prefix="refattn",  # 你也可以用 view_{i} 之类的名字
                    )
                except Exception as e:
                    print("[DBG] save debug images failed:", repr(e))

            ref_image = reference_image if reference_image is not None else image
            ref_init = self.image_processor.preprocess(
                ref_image,
                height=height,
                width=width,
                crops_coords=crops_coords,
                resize_mode=resize_mode,
            )
            ref_init = ref_init.to(dtype=torch.float32)

            device = self._execution_device
            vae_dtype = next(self.vae.parameters()).dtype
            ref_init = ref_init.to(device=device, dtype=vae_dtype)

            with torch.inference_mode():
                ref_lat = self._encode_vae_image(ref_init, generator=None)
                neg_lat = self._encode_vae_image(
                    torch.zeros_like(ref_init), generator=None
                )

            is_cfg = float(guidance_scale) > 1.0
            cond_lat = torch.cat([neg_lat, ref_lat], dim=0) if is_cfg else ref_lat

            if hasattr(self.unet, "set_cond_lat"):
                self.unet.set_cond_lat(cond_lat, is_cfg_guidance=is_cfg)

            if hasattr(self.unet, "set_inpaint_extra"):
                # Keep the write pass reference-only. For the inpaint UNet's extra
                # 5 channels, use a zero mask together with the full reference-image latent,
                # instead of the current view's mask/masked-image latents.
                ref_mask_lat = torch.zeros(
                    (ref_lat.shape[0], 1, ref_lat.shape[-2], ref_lat.shape[-1]),
                    device=ref_lat.device,
                    dtype=ref_lat.dtype,
                )
                extra = torch.cat([ref_mask_lat, ref_lat], dim=1)

                if extra.shape[0] != cond_lat.shape[0]:
                    if extra.shape[0] == 1:
                        extra = extra.repeat(cond_lat.shape[0], 1, 1, 1)
                    else:
                        extra = extra[: cond_lat.shape[0]]

                self.unet.set_inpaint_extra(extra)

        # ---------------- IP-Adapter prompt token concat ----------------
        if (image_prompt_embeds is None) ^ (negative_image_prompt_embeds is None):
            raise ValueError(
                "image_prompt_embeds and negative_image_prompt_embeds "
                "must be provided together."
            )

        if image_prompt_embeds is not None:
            if prompt_embeds is None or negative_prompt_embeds is None:
                do_classifier_free_guidance = float(guidance_scale) > 1.0
                text_encoder_lora_scale = None

                if isinstance(cross_attention_kwargs, dict):
                    text_encoder_lora_scale = cross_attention_kwargs.get("scale", None)

                prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                    prompt,
                    self._execution_device,
                    num_images_per_prompt,
                    do_classifier_free_guidance,
                    negative_prompt,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    lora_scale=text_encoder_lora_scale,
                    clip_skip=clip_skip,
                )

            to_dtype = prompt_embeds.dtype
            ip_pos = image_prompt_embeds.to(to_dtype)
            ip_neg = negative_image_prompt_embeds.to(to_dtype)

            if ip_pos.shape[0] != prompt_embeds.shape[0]:
                ip_pos = (
                    ip_pos.repeat(prompt_embeds.shape[0], 1, 1)
                    if ip_pos.shape[0] == 1
                    else ip_pos[: prompt_embeds.shape[0]]
                )

            if ip_neg.shape[0] != negative_prompt_embeds.shape[0]:
                ip_neg = (
                    ip_neg.repeat(negative_prompt_embeds.shape[0], 1, 1)
                    if ip_neg.shape[0] == 1
                    else ip_neg[: negative_prompt_embeds.shape[0]]
                )

            prompt_embeds = torch.cat([prompt_embeds, ip_pos], dim=1)
            negative_prompt_embeds = torch.cat(
                [negative_prompt_embeds, ip_neg], dim=1
            )

        # If prompt_embeds are provided/constructed, do not pass prompt strings
        # (diffusers will error).
        if prompt_embeds is not None:
            prompt = None
        if negative_prompt_embeds is not None:
            negative_prompt = None

        # RefOnlyNoisedUNet expects cross_attention_kwargs to be a dict
        # when reference_image is used.
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}
        else:
            cross_attention_kwargs = dict(cross_attention_kwargs)

        if reference_image is not None:
            cross_attention_kwargs["cond_lat"] = cond_lat
            cross_attention_kwargs["is_cfg_guidance"] = is_cfg

        try:
            return super().__call__(
                prompt=prompt,
                image=image,
                mask_image=mask_image,
                control_image=control_image,
                height=height,
                width=width,
                padding_mask_crop=padding_mask_crop,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                eta=eta,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                ip_adapter_image=ip_adapter_image,
                output_type=output_type,
                return_dict=return_dict,
                cross_attention_kwargs=cross_attention_kwargs,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                guess_mode=guess_mode,
                control_guidance_start=control_guidance_start,
                control_guidance_end=control_guidance_end,
                clip_skip=clip_skip,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                **kwargs,
            )
        finally:
            if hasattr(self.unet, "clear_reference_conditioning"):
                self.unet.clear_reference_conditioning(reset_progress=True)