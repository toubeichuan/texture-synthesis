"""
Trainable utilities for Easi-Tex reference attention.

Why this file exists:
- `lib/reference_attention.py` focuses on inference behavior (write/read reference cache).
- Fine-tuning usually needs trainable parameters while keeping UNet mostly frozen.

This module adds a trainable attention processor that is API-compatible with the
existing reference-attention flow and can be attached to the same UNet.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, Optional

import torch
import torch.nn as nn

from lib import reference_attention as ref_attn


def _unwrap_unet(unet: nn.Module) -> nn.Module:
    if isinstance(unet, ref_attn.RefOnlyNoisedUNet):
        return unet.unet
    return unet


def _infer_hidden_size(unet: nn.Module, processor_name: str) -> int:
    """
    Infer hidden size for a UNet attention processor from diffusers naming convention.
    """
    name = str(processor_name)
    block_out_channels = list(getattr(unet.config, "block_out_channels", []))
    if not block_out_channels:
        raise ValueError("UNet config has no `block_out_channels`; cannot infer hidden size.")

    if name.startswith("mid_block"):
        return int(block_out_channels[-1])

    if name.startswith("up_blocks"):
        block_id = int(name.split(".")[1])
        return int(list(reversed(block_out_channels))[block_id])

    if name.startswith("down_blocks"):
        block_id = int(name.split(".")[1])
        return int(block_out_channels[block_id])

    # Fallback for uncommon names.
    return int(block_out_channels[0])


class RefAttnLoRAAdapter(nn.Module):
    """
    Low-rank adapter applied to reference tokens before read-time concatenation.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        rank: int = 8,
        lora_alpha: float = 8.0,
        dropout: float = 0.0,
        init_ref_scale: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_scale = self.lora_alpha / float(max(1, self.rank))
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.ref_scale = nn.Parameter(torch.tensor(float(init_ref_scale), dtype=torch.float32))

        if self.rank > 0:
            self.down = nn.Linear(self.hidden_size, self.rank, bias=False)
            self.up = nn.Linear(self.rank, self.hidden_size, bias=False)
            nn.init.normal_(self.down.weight, mean=0.0, std=1.0 / float(self.rank))
            nn.init.zeros_(self.up.weight)
        else:
            self.down = None
            self.up = None

    def forward(self, ref_hidden_states: torch.Tensor) -> torch.Tensor:
        ref_hidden_states = ref_hidden_states * self.ref_scale.to(
            device=ref_hidden_states.device,
            dtype=ref_hidden_states.dtype,
        )

        if self.rank <= 0 or self.down is None or self.up is None:
            return ref_hidden_states

        delta = self.up(self.down(self.dropout(ref_hidden_states)))
        return ref_hidden_states + (self.lora_scale * delta)


class TrainableReferenceOnlyAttnProc(nn.Module):
    """
    A trainable variant of `ReferenceOnlyAttnProc`.

    Behavior:
    - mode="w": cache encoder states into ref_dict
    - mode="r": read cached states and concatenate after trainable adaptation
    - mode="off": passthrough to chained processor
    """

    def __init__(
        self,
        chained_proc: Any,
        *,
        enabled: bool,
        name: str,
        hidden_size: Optional[int],
        rank: int = 8,
        lora_alpha: float = 8.0,
        dropout: float = 0.0,
        init_ref_scale: float = 1.0,
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.chained_proc = chained_proc
        self.name = str(name)
        self.adapter: Optional[RefAttnLoRAAdapter]
        if self.enabled and hidden_size is not None:
            self.adapter = RefAttnLoRAAdapter(
                hidden_size=int(hidden_size),
                rank=int(rank),
                lora_alpha=float(lora_alpha),
                dropout=float(dropout),
                init_ref_scale=float(init_ref_scale),
            )
        else:
            self.adapter = None

    def _passthrough(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        **kwargs,
    ):
        return self.chained_proc(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            **kwargs,
        )

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        **kwargs,
    ):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # Preferred path: kwargs injected by RefOnlyNoisedUNet.
        if "mode" in kwargs and "ref_dict" in kwargs and "is_cfg_guidance" in kwargs:
            mode = kwargs.pop("mode")
            ref_dict = kwargs.pop("ref_dict")
            is_cfg_guidance = kwargs.pop("is_cfg_guidance")
        else:
            # Backward compatibility path with global context.
            mode = ref_attn._STATE.mode
            ref_dict = ref_attn._STATE.ref_dict
            is_cfg_guidance = ref_attn._STATE.is_cfg_guidance
            if mode == "off" or ref_dict is None:
                return self._passthrough(
                    attn,
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    temb=temb,
                    **kwargs,
                )

        if not self.enabled or mode == "off" or ref_dict is None:
            return self._passthrough(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                temb=temb,
                **kwargs,
            )

        mode = str(mode)
        is_cfg_guidance = bool(is_cfg_guidance)

        # Keep the same CFG behavior as existing implementation.
        if is_cfg_guidance:
            res0 = self.chained_proc(
                attn,
                hidden_states[:1],
                encoder_hidden_states=encoder_hidden_states[:1],
                attention_mask=attention_mask,
                temb=temb,
                **kwargs,
            )
            hidden_states = hidden_states[1:]
            encoder_hidden_states = encoder_hidden_states[1:]
        else:
            res0 = None

        if mode == "w":
            ref_dict[self.name] = encoder_hidden_states
        elif mode == "r":
            ref = ref_dict.pop(self.name, None)
            if ref is not None:
                if self.adapter is not None:
                    ref = self.adapter(ref)
                ref = ref.to(device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
                encoder_hidden_states = torch.cat([encoder_hidden_states, ref], dim=1)
        elif mode == "m":
            ref = ref_dict.get(self.name, None)
            if ref is not None:
                if self.adapter is not None:
                    ref = self.adapter(ref)
                ref = ref.to(device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
                encoder_hidden_states = torch.cat([encoder_hidden_states, ref], dim=1)
        else:
            raise ValueError(f"Unsupported reference-attention mode: {mode}")

        out = self.chained_proc(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            **kwargs,
        )
        if res0 is not None:
            out = torch.cat([res0, out], dim=0)
        return out


def enable_trainable_reference_attention(
    unet: nn.Module,
    *,
    rank: int = 8,
    lora_alpha: float = 8.0,
    dropout: float = 0.0,
    init_ref_scale: float = 1.0,
) -> nn.Module:
    """
    Replace UNet self-attention processors with trainable reference-attention processors.
    """
    core_unet = _unwrap_unet(unet)
    if not hasattr(core_unet, "attn_processors") or not hasattr(core_unet, "set_attn_processor"):
        raise ValueError("UNet must expose `attn_processors` and `set_attn_processor`.")

    device = next(core_unet.parameters()).device
    dtype = next(core_unet.parameters()).dtype

    new_procs: Dict[str, Any] = {}
    for name, proc in core_unet.attn_processors.items():
        proc_name = str(name)
        is_self_attn = ("attn1" in proc_name) and ("attn2" not in proc_name)

        if isinstance(proc, TrainableReferenceOnlyAttnProc):
            wrapped = proc
        else:
            # If already wrapped by non-trainable ReferenceOnlyAttnProc, unwrap it first.
            base_proc = getattr(proc, "chained_proc", proc)
            hidden_size = _infer_hidden_size(core_unet, proc_name) if is_self_attn else None
            wrapped = TrainableReferenceOnlyAttnProc(
                base_proc,
                enabled=is_self_attn,
                name=proc_name,
                hidden_size=hidden_size,
                rank=int(rank),
                lora_alpha=float(lora_alpha),
                dropout=float(dropout),
                init_ref_scale=float(init_ref_scale),
            )

        if isinstance(wrapped, nn.Module):
            wrapped.to(device=device, dtype=dtype)
        new_procs[name] = wrapped

    core_unet.set_attn_processor(new_procs)
    # Keep compatibility with existing `ensure_ref_unet_wrapped` calls in the pipeline.
    # If this flag is not set, the old helper may re-wrap processors and overwrite
    # the trainable processors we just installed.
    setattr(core_unet, "_easitex_ref_attn_enabled", True)
    setattr(core_unet, "_easitex_ref_attn_trainable", True)
    return unet


def ensure_trainable_ref_unet_wrapped(
    pipe,
    *,
    rank: int = 8,
    lora_alpha: float = 8.0,
    dropout: float = 0.0,
    init_ref_scale: float = 1.0,
) -> None:
    """
    Ensure both:
    1) trainable ref-attn processors are installed
    2) UNet is wrapped by RefOnlyNoisedUNet (write/read forward behavior)
    """
    if not hasattr(pipe, "unet"):
        raise ValueError("Pipeline has no `unet` attribute.")
    if not hasattr(pipe, "scheduler"):
        raise ValueError("Pipeline has no `scheduler` attribute.")

    if isinstance(pipe.unet, ref_attn.RefOnlyNoisedUNet):
        enable_trainable_reference_attention(
            pipe.unet.unet,
            rank=rank,
            lora_alpha=lora_alpha,
            dropout=dropout,
            init_ref_scale=init_ref_scale,
        )
        return

    enable_trainable_reference_attention(
        pipe.unet,
        rank=rank,
        lora_alpha=lora_alpha,
        dropout=dropout,
        init_ref_scale=init_ref_scale,
    )
    pipe.unet = ref_attn.RefOnlyNoisedUNet(pipe.unet, pipe.scheduler)


def get_trainable_refattn_processors(unet: nn.Module) -> Dict[str, TrainableReferenceOnlyAttnProc]:
    core_unet = _unwrap_unet(unet)
    out: Dict[str, TrainableReferenceOnlyAttnProc] = {}
    for name, proc in core_unet.attn_processors.items():
        if isinstance(proc, TrainableReferenceOnlyAttnProc) and proc.adapter is not None:
            out[str(name)] = proc
    return out


def iter_trainable_refattn_parameters(unet: nn.Module) -> Iterator[nn.Parameter]:
    for proc in get_trainable_refattn_processors(unet).values():
        yield from proc.adapter.parameters()


def freeze_unet_except_refattn(unet: nn.Module) -> None:
    core_unet = _unwrap_unet(unet)
    for p in core_unet.parameters():
        p.requires_grad = False
    for p in iter_trainable_refattn_parameters(core_unet):
        p.requires_grad = True


def get_trainable_refattn_state_dict(unet: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    state: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, proc in get_trainable_refattn_processors(unet).items():
        state[name] = proc.adapter.state_dict()
    return state


def save_trainable_refattn(unet: nn.Module, save_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    payload = {
        "format": "easitex_refattn_finetune_v1",
        "state_dict": get_trainable_refattn_state_dict(unet),
    }
    torch.save(payload, save_path)


def load_trainable_refattn(
    unet: nn.Module,
    load_path: str,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    payload = torch.load(load_path, map_location=map_location)
    state_dict = payload.get("state_dict", payload)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Invalid ref-attn checkpoint format: {load_path}")

    procs = get_trainable_refattn_processors(unet)
    loaded, missing, unexpected = [], [], []
    for name, proc in procs.items():
        if name not in state_dict:
            missing.append(name)
            continue
        proc.adapter.load_state_dict(state_dict[name], strict=strict)
        loaded.append(name)

    for name in state_dict.keys():
        if name not in procs:
            unexpected.append(name)

    if strict and (missing or unexpected):
        raise RuntimeError(
            "Ref-attn adapter load mismatch. "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    return {
        "loaded": loaded,
        "missing": missing,
        "unexpected": unexpected,
    }
