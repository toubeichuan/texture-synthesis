import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from diffusers import ControlNetModel, DDIMScheduler

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from lib.pipeline_controlnet_inpaint_refattn import StableDiffusionControlNetInpaintRefAttnPipeline
from lib.reference_attention_finetune import (
    ensure_trainable_ref_unet_wrapped,
    freeze_unet_except_refattn,
    iter_trainable_refattn_parameters,
    load_trainable_refattn,
    save_trainable_refattn,
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _pick_local_or_hf_id(local_dir: str, hf_id: str):
    if local_dir and os.path.isdir(local_dir):
        return local_dir, True
    return hf_id, False


def _hf_cache_snapshot_dir(model_cache_dir: str) -> str:
    try:
        refs_main = os.path.join(model_cache_dir, "refs", "main")
        if os.path.isfile(refs_main):
            with open(refs_main, "r", encoding="utf-8") as f:
                rev = f.read().strip()
            snap = os.path.join(model_cache_dir, "snapshots", rev)
            if os.path.isdir(snap):
                return snap
        snaps_dir = os.path.join(model_cache_dir, "snapshots")
        if os.path.isdir(snaps_dir):
            snaps = [d for d in os.listdir(snaps_dir) if os.path.isdir(os.path.join(snaps_dir, d))]
            if len(snaps) == 1:
                return os.path.join(snaps_dir, snaps[0])
    except Exception:
        return ""
    return ""


def resolve_path(base_dir: str, p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(base_dir, p))


@dataclass
class TrainExample:
    image: str
    mask: str
    control: str
    reference: str
    target: str
    prompt: str


class RefAttnJsonlDataset(Dataset):
    """
    Each jsonl line should contain keys:
      - image: path to init render RGB
      - mask: path to inpaint mask (white=inpaint, black=keep)
      - control: path to control image (depth/canny RGB or grayscale)
      - reference: path to reference image for ref-attn
      - target: path to target RGB image
      - prompt: text prompt
    """

    def __init__(self, jsonl_path: str):
        if not os.path.isfile(jsonl_path):
            raise FileNotFoundError(f"train_jsonl not found: {jsonl_path}")
        self.base_dir = os.path.dirname(os.path.abspath(jsonl_path))
        self.examples: List[TrainExample] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for ln, raw in enumerate(f, start=1):
                raw = raw.strip()
                if len(raw) == 0:
                    continue
                rec = json.loads(raw)
                required = ["image", "mask", "control", "reference", "target", "prompt"]
                miss = [k for k in required if k not in rec]
                if miss:
                    raise ValueError(f"{jsonl_path}:{ln} missing keys: {miss}")
                ex = TrainExample(
                    image=resolve_path(self.base_dir, rec["image"]),
                    mask=resolve_path(self.base_dir, rec["mask"]),
                    control=resolve_path(self.base_dir, rec["control"]),
                    reference=resolve_path(self.base_dir, rec["reference"]),
                    target=resolve_path(self.base_dir, rec["target"]),
                    prompt=str(rec["prompt"]),
                )
                self.examples.append(ex)
        if len(self.examples) == 0:
            raise ValueError(f"Empty dataset: {jsonl_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index: int):
        ex = self.examples[index]
        with Image.open(ex.image) as im:
            image = im.convert("RGB")
        with Image.open(ex.mask) as im:
            mask = im.convert("L")
        with Image.open(ex.control) as im:
            control = im.convert("RGB")
        with Image.open(ex.reference) as im:
            reference = im.convert("RGB")
        with Image.open(ex.target) as im:
            target = im.convert("RGB")
        return {
            "image": image,
            "mask": mask,
            "control": control,
            "reference": reference,
            "target": target,
            "prompt": ex.prompt,
        }


def collate_fn(batch: List[Dict]):
    out = {k: [] for k in batch[0].keys()}
    for item in batch:
        for k, v in item.items():
            out[k].append(v)
    return out


def load_training_pipeline(args, device: torch.device, torch_dtype: torch.dtype):
    hub_root = os.environ.get("HF_HUB_CACHE_ROOT", "/data/23009960/hub")
    default_base = _hf_cache_snapshot_dir(
        os.path.join(hub_root, "models--runwayml--stable-diffusion-inpainting")
    )
    base_local = args.sd15_inpaint_model_dir or default_base
    base_id, base_local_only = _pick_local_or_hf_id(base_local, "runwayml/stable-diffusion-inpainting")

    if args.controlnet_cond == "depth":
        default_cn = _hf_cache_snapshot_dir(
            os.path.join(hub_root, "models--lllyasviel--control_v11f1p_sd15_depth")
        )
        cn_local = args.sd15_controlnet_depth_dir or default_cn
        cn_id, cn_local_only = _pick_local_or_hf_id(cn_local, "lllyasviel/control_v11f1p_sd15_depth")
    else:
        default_cn = _hf_cache_snapshot_dir(
            os.path.join(hub_root, "models--lllyasviel--control_v11p_sd15_canny")
        )
        cn_local = args.sd15_controlnet_canny_dir or default_cn
        cn_id, cn_local_only = _pick_local_or_hf_id(cn_local, "lllyasviel/control_v11p_sd15_canny")

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
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    except Exception:
        pass
    return pipe


def parse_args():
    parser = argparse.ArgumentParser(description="Train only Easi-Tex reference-attention adapters.")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--controlnet_cond", type=str, default="canny", choices=["canny", "depth"])
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--resolution", type=int, default=768)

    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--refattn_rank", type=int, default=8)
    parser.add_argument("--refattn_lora_alpha", type=float, default=8.0)
    parser.add_argument("--refattn_dropout", type=float, default=0.0)
    parser.add_argument("--refattn_init_ref_scale", type=float, default=1.0)
    parser.add_argument("--resume_refattn_ckpt", type=str, default="")
    parser.add_argument("--resume_refattn_strict", type=str2bool, default=True)

    parser.add_argument("--sd15_inpaint_model_dir", type=str, default="")
    parser.add_argument("--sd15_controlnet_canny_dir", type=str, default="")
    parser.add_argument("--sd15_controlnet_depth_dir", type=str, default="")

    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("REF_ATTN_PROGRESS", "0")
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"=> device={device}, dtype={torch_dtype}")

    pipe = load_training_pipeline(args, device=device, torch_dtype=torch_dtype)

    ensure_trainable_ref_unet_wrapped(
        pipe,
        rank=int(args.refattn_rank),
        lora_alpha=float(args.refattn_lora_alpha),
        dropout=float(args.refattn_dropout),
        init_ref_scale=float(args.refattn_init_ref_scale),
    )
    if hasattr(pipe.unet, "_enable_progress"):
        pipe.unet._enable_progress = False

    if args.resume_refattn_ckpt:
        load_info = load_trainable_refattn(
            pipe.unet,
            args.resume_refattn_ckpt,
            strict=bool(args.resume_refattn_strict),
        )
        print(
            "=> resumed ref-attn ckpt: "
            f"{args.resume_refattn_ckpt} | "
            f"loaded={len(load_info['loaded'])}, "
            f"missing={len(load_info['missing'])}, "
            f"unexpected={len(load_info['unexpected'])}"
        )

    freeze_unet_except_refattn(pipe.unet)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.controlnet.requires_grad_(False)

    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.controlnet.eval()
    pipe.unet.train()

    trainable_params = [p for p in iter_trainable_refattn_parameters(pipe.unet) if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable ref-attn parameters found.")
    total_params = sum(p.numel() for p in trainable_params)
    print(f"=> trainable ref-attn params: {total_params}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    dataset = RefAttnJsonlDataset(args.train_jsonl)
    dataloader = DataLoader(
        dataset,
        batch_size=int(args.train_batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=collate_fn,
        drop_last=True,
    )

    vae_dtype = next(pipe.vae.parameters()).dtype
    unet_dtype = next(pipe.unet.parameters()).dtype
    text_dtype = next(pipe.text_encoder.parameters()).dtype
    control_dtype = next(pipe.controlnet.parameters()).dtype

    global_step = 0
    accum_steps = max(1, int(args.gradient_accumulation_steps))
    max_train_steps = int(args.max_train_steps) if int(args.max_train_steps) > 0 else None
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(int(args.num_train_epochs)):
        for step, batch in enumerate(dataloader):
            bsz = len(batch["prompt"])
            h = int(args.resolution)
            w = int(args.resolution)

            image = pipe.image_processor.preprocess(batch["image"], height=h, width=w).to(device=device, dtype=torch.float32)
            target_image = pipe.image_processor.preprocess(batch["target"], height=h, width=w).to(device=device, dtype=torch.float32)
            reference_image = pipe.image_processor.preprocess(batch["reference"], height=h, width=w).to(device=device, dtype=torch.float32)
            mask = pipe.mask_processor.preprocess(batch["mask"], height=h, width=w).to(device=device, dtype=torch.float32)
            mask = (mask > 0.5).float()
            masked_image = image * (mask < 0.5)

            with torch.no_grad():
                prompt_embeds, _ = pipe.encode_prompt(
                    batch["prompt"],
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt=None,
                )
                prompt_embeds = prompt_embeds.to(device=device, dtype=text_dtype)

                control_image = pipe.prepare_control_image(
                    image=batch["control"],
                    width=w,
                    height=h,
                    batch_size=bsz,
                    num_images_per_prompt=1,
                    device=device,
                    dtype=control_dtype,
                    crops_coords=None,
                    resize_mode="default",
                    do_classifier_free_guidance=False,
                    guess_mode=False,
                )

                target_latents = pipe._encode_vae_image(target_image.to(device=device, dtype=vae_dtype), generator=None)
                reference_latents = pipe._encode_vae_image(reference_image.to(device=device, dtype=vae_dtype), generator=None)
                masked_image_latents = pipe._encode_vae_image(masked_image.to(device=device, dtype=vae_dtype), generator=None)
                mask_latents = F.interpolate(mask.to(device=device, dtype=vae_dtype), size=masked_image_latents.shape[-2:], mode="nearest")

            if hasattr(pipe.unet, "set_cond_lat"):
                pipe.unet.set_cond_lat(reference_latents, is_cfg_guidance=False)
            if hasattr(pipe.unet, "set_inpaint_extra"):
                pipe.unet.set_inpaint_extra(torch.cat([mask_latents, masked_image_latents], dim=1))

            noise = torch.randn_like(target_latents)
            timesteps = torch.randint(
                0,
                pipe.scheduler.config.num_train_timesteps,
                (bsz,),
                device=device,
                dtype=torch.long,
            )
            noisy_latents = pipe.scheduler.add_noise(target_latents, noise, timesteps)
            latent_model_input = pipe.scheduler.scale_model_input(noisy_latents, timesteps)

            with torch.no_grad():
                down_block_res_samples, mid_block_res_sample = pipe.controlnet(
                    latent_model_input.to(dtype=control_dtype),
                    timesteps,
                    encoder_hidden_states=prompt_embeds.to(dtype=control_dtype),
                    controlnet_cond=control_image,
                    conditioning_scale=float(args.controlnet_conditioning_scale),
                    guess_mode=False,
                    return_dict=False,
                )

            unet_input = torch.cat(
                [
                    latent_model_input.to(dtype=unet_dtype),
                    mask_latents.to(dtype=unet_dtype),
                    masked_image_latents.to(dtype=unet_dtype),
                ],
                dim=1,
            )

            noise_pred = pipe.unet(
                unet_input,
                timesteps,
                encoder_hidden_states=prompt_embeds.to(dtype=unet_dtype),
                cross_attention_kwargs={},
                down_block_additional_residuals=[d.to(dtype=unet_dtype) for d in down_block_res_samples],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=unet_dtype),
                return_dict=False,
            )[0]

            prediction_type = getattr(pipe.scheduler.config, "prediction_type", "epsilon")
            if prediction_type == "epsilon":
                target = noise
            elif prediction_type == "v_prediction":
                target = pipe.scheduler.get_velocity(target_latents, noise, timesteps)
            else:
                raise ValueError(f"Unsupported prediction_type: {prediction_type}")

            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
            (loss / accum_steps).backward()

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, float(args.max_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % int(args.log_every) == 0:
                    print(f"=> step={global_step} loss={float(loss.detach().cpu().item()):.6f}")

                if global_step % int(args.save_every) == 0:
                    ckpt_path = os.path.join(args.output_dir, f"refattn_step_{global_step}.pt")
                    save_trainable_refattn(pipe.unet, ckpt_path)
                    print(f"=> saved checkpoint: {ckpt_path}")

                if max_train_steps is not None and global_step >= max_train_steps:
                    break

        if max_train_steps is not None and global_step >= max_train_steps:
            break

    final_ckpt = os.path.join(args.output_dir, "refattn_final.pt")
    save_trainable_refattn(pipe.unet, final_ckpt)
    print(f"=> training done, final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
