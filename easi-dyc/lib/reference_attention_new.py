"""
Reference Attention (minimal port inspired by MeshGen/zero123pp)

Goal:
- During denoising, inject K/V from the reference image's *self-attention* into the current denoising step,
  to improve consistency with the reference image.

Implementation approach (diffusers-friendly, no pipeline changes required):
- Wrap UNet's self-attention processors (attn1) with a processor that can:
  - mode="w": write (cache) encoder_hidden_states (for self-attn: hidden_states) into a ref_dict
  - mode="r": read cached states and concatenate on the sequence dimension
- Wrap UNet forward:
  - For each timestep, first run a dedicated 4-channel writer UNet on a *noised reference latent*
    in write mode to populate ref_dict
  - Then run the original reader UNet on current sample latent in read mode to consume cached ref_dict

Note:
- We avoid relying on `cross_attention_kwargs` since some bundled diffusers versions don't forward it.
  Instead, we use a small module-level context (mode/ref_dict/is_cfg_guidance) set by the UNet wrapper.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional
import os

import torch
import torch.nn as nn


@dataclass
class _RefAttnState:
    mode: str = "off"  # "w" | "r" | "m" | "off"
    ref_dict: Optional[Dict[str, torch.Tensor]] = None
    is_cfg_guidance: bool = False


def _should_split_cfg_reference() -> bool:
    """
    在 easi-tex 当前这条显式 CFG 路径里，processor 侧最好把 uncond/cond 分开：
    - write pass 只把 conditional half 写入 ref cache
    - read pass 只让 conditional half 读取 ref cache

    否则 unconditional 分支也会被 reference 特征污染，容易把 generate 区域
    拉向发黑/发脏的结果。

    如需关闭 split，可设置：
        EASITEX_REF_ATTN_SPLIT_CFG=0
    """
    return os.environ.get("EASITEX_REF_ATTN_SPLIT_CFG", "1") != "0"


_STATE = _RefAttnState()


@contextmanager
def ref_attention_context(*, mode: str, ref_dict: Dict[str, torch.Tensor], is_cfg_guidance: bool):
    prev = (_STATE.mode, _STATE.ref_dict, _STATE.is_cfg_guidance)
    _STATE.mode = mode
    _STATE.ref_dict = ref_dict
    _STATE.is_cfg_guidance = is_cfg_guidance
    try:
        yield
    finally:
        _STATE.mode, _STATE.ref_dict, _STATE.is_cfg_guidance = prev


class ReferenceOnlyAttnProc(nn.Module):
    """
    A wrapper for diffusers attention processors (e.g., AttnProcessor2_0, XFormersAttnProcessor, etc.).

    It only activates when enabled=True and global ref_attention_context is set (mode != "off").
    """

    def __init__(self, chained_proc: Any, enabled: bool = False, name: Optional[str] = None):
        super().__init__()
        self.enabled = bool(enabled)
        self.chained_proc = chained_proc
        self.name = name or "attn"

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
        # Keep behavior identical when disabled
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # Extract mode/ref_dict/is_cfg_guidance from kwargs (diffusers expands cross_attention_kwargs into kwargs)
        # Fallback to global context if not in kwargs (for backward compatibility)
        # CRITICAL: Use the module-level _STATE directly (same file), don't re-import to avoid module duplication
        has_refattn_kwargs = any(
            k in kwargs for k in ("mode", "ref_dict", "is_cfg_guidance")
        )
        if has_refattn_kwargs:
            mode = kwargs.pop("mode", _STATE.mode)
            ref_dict = kwargs.pop("ref_dict", _STATE.ref_dict)
            is_cfg_guidance = kwargs.pop("is_cfg_guidance", _STATE.is_cfg_guidance)
            source = "kwargs"
        else:
            # Fallback to global context (should not happen if cross_attention_kwargs is properly set)
            # Use module-level _STATE directly (defined at top of this file)
            mode = _STATE.mode
            ref_dict = _STATE.ref_dict
            is_cfg_guidance = _STATE.is_cfg_guidance
            source = "STATE"
            if mode == "off" or ref_dict is None:
                # No reference attention active, pass through
                return self.chained_proc(
                    attn,
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    temb=temb,
                    **kwargs,
                )

        # Debug: only for first few invocations to avoid spam
        debug_attn = False
        if hasattr(self, '_debug_step_count'):
            self._debug_step_count = getattr(self, '_debug_step_count', 0) + 1
            debug_attn = self._debug_step_count <= 3
        else:
            self._debug_step_count = 1
            debug_attn = True

        if not self.enabled or mode == "off" or ref_dict is None:
            return self.chained_proc(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                temb=temb,
                **kwargs,
            )

        mode = str(mode)
        is_cfg_guidance = bool(is_cfg_guidance)

        #  1：进入 write/read 前，只对 attn1（self.enabled）打一次
        if debug_attn and self.enabled:
            hs_sh = tuple(hidden_states.shape)
            ehs_sh = tuple(encoder_hidden_states.shape)
            print(f"[ATTN] {self.name} ENABLED mode={mode} src={source} "
                  f"hs={hs_sh} ehs={ehs_sh} dict_len={len(ref_dict)} dict_id={id(ref_dict)}")
        # CFG 污染检查：batch 与 is_cfg 是否匹配
        if debug_attn and self.enabled and mode in ("w", "r"):
            print(f"[CFGCHK] {self.name} batch={hidden_states.shape[0]} is_cfg={is_cfg_guidance}")
        # In CFG, UNet is called with a batch of size 2*B laid out as
        # [uncond..., cond...]. Keep the unconditional half isolated so
        # reference write/read only affects the conditional half.
        if mode in ("w", "r") and is_cfg_guidance and self.enabled and _should_split_cfg_reference():
            cfg_batch = hidden_states.shape[0] // 2 if hidden_states.shape[0] % 2 == 0 else 1
            res0 = self.chained_proc(
                attn,
                hidden_states[:cfg_batch],
                encoder_hidden_states=encoder_hidden_states[:cfg_batch],
                attention_mask=attention_mask,
                temb=temb,
                **kwargs,
            )
            hidden_states = hidden_states[cfg_batch:]
            encoder_hidden_states = encoder_hidden_states[cfg_batch:]
        else:
            res0 = None

        if self.enabled:
            if mode == "w":
                ref_dict[self.name] = encoder_hidden_states
                if debug_attn:
                    print(f"[W] {self.name} stored shape={tuple(encoder_hidden_states.shape)} dict_len={len(ref_dict)}")
            elif mode == "r":
                if self.name not in ref_dict:
                    print(f"[R][WARN] {self.name} missing in ref_dict! dict_keys_sample={list(ref_dict.keys())[:5]}")
                else:
                    before = tuple(encoder_hidden_states.shape)
                    ref = ref_dict.pop(self.name)
                    encoder_hidden_states = torch.cat([encoder_hidden_states, ref], dim=1)
                    after = tuple(encoder_hidden_states.shape)
                    if debug_attn:
                        print(f"[R] {self.name} cat {before} + {tuple(ref.shape)} -> {after}  dict_len={len(ref_dict)}")
            elif mode == "m":
                encoder_hidden_states = torch.cat(
                    [encoder_hidden_states, ref_dict[self.name]], dim=1
                )
            else:
                assert False, mode

        res = self.chained_proc(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            **kwargs,
        )

        if res0 is not None:
            res = torch.cat([res0, res], dim=0)
        return res


def enable_reference_attention(unet: nn.Module) -> nn.Module:
    """
    Patch UNet attention processors in-place:
    - Wrap self-attention (attn1) processors with ReferenceOnlyAttnProc(enabled=True)
    - Keep other processors unchanged
    """
    if not hasattr(unet, "attn_processors"):
        raise ValueError("UNet does not expose `attn_processors`; cannot enable reference attention.")
    if not hasattr(unet, "set_attn_processor"):
        raise ValueError("UNet does not expose `set_attn_processor`; cannot enable reference attention.")

    already_wrapped = True
    for name, proc in unet.attn_processors.items():
        n = str(name)
        should_enable = ("attn1" in n) and ("attn2" not in n)
        if not isinstance(proc, ReferenceOnlyAttnProc):
            already_wrapped = False
            break
        if bool(proc.enabled) != should_enable:
            already_wrapped = False
            break
    if already_wrapped and getattr(unet, "_easitex_ref_attn_enabled", False):
        return unet

    new_procs = {}
    for name, proc in unet.attn_processors.items():
        # Diffusers convention: attn1 = self-attention, attn2 = cross-attention
        # Some diffusers versions use keys like "...attn1.processor", some may differ slightly;
        # enabling on any "attn1" is the safest to make sure reference attention actually activates.
        n = str(name)
        enabled = ("attn1" in n) and ("attn2" not in n)
        base_proc = proc.chained_proc if isinstance(proc, ReferenceOnlyAttnProc) else proc
        new_procs[name] = ReferenceOnlyAttnProc(base_proc, enabled=enabled, name=name)

    unet.set_attn_processor(new_procs)
    setattr(unet, "_easitex_ref_attn_enabled", True)
    return unet


class RefOnlyNoisedUNet(nn.Module):
    """
    A UNet wrapper that, for each forward call (timestep), performs:
      1) run an explicit 4-channel writer UNet on a *noised reference latent* in
         write mode to populate ref_dict
      2) run the original reader UNet on the real sample latent in read mode to
         consume ref_dict

    The reference latent (cond_lat) must be set via `set_cond_lat(...)` before sampling.
    """

    def __init__(self, unet: nn.Module, scheduler):
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler
        # Clone the reader UNet into an explicit 4-channel writer so the write
        # pass behaves like a normal SD latent branch while keeping attention
        # topology and weights maximally aligned with the 9-channel reader.
        self.writer_unet = self._build_reference_writer(unet)
        self._cond_lat: Optional[torch.Tensor] = None
        # Legacy field kept for compatibility with older callers; the optimized
        # write path no longer consumes inpaint extra channels.
        self._cond_extra: Optional[torch.Tensor] = None
        self._is_cfg_guidance: bool = False
        # progress tracking for write/read passes
        self._step_idx: int = 0
        self._total_steps: Optional[int] = None
        # allow turning off progress bar via env
        self._enable_progress: bool = os.environ.get("REF_ATTN_PROGRESS", "1") != "0"

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)

    def _uses_inpaint_extra(self) -> bool:
        conv_in = getattr(self.unet, "conv_in", None)
        return isinstance(conv_in, nn.Conv2d) and int(conv_in.in_channels) > 4

    def set_cond_lat(self, cond_lat: torch.Tensor, *, is_cfg_guidance: bool):
        self._cond_lat = cond_lat
        self._is_cfg_guidance = bool(is_cfg_guidance)
        if not self._uses_inpaint_extra():
            self._cond_extra = None
        self.ensure_writer_alignment()

    def set_inpaint_extra(self, cond_extra: Optional[torch.Tensor]):
        """
        Legacy no-op state for backward compatibility.
        """
        if not self._uses_inpaint_extra():
            self._cond_extra = None
            return
        self._cond_extra = cond_extra

    def clear_reference_conditioning(self, *, reset_progress: bool = True):
        self._cond_lat = None
        self._cond_extra = None
        self._is_cfg_guidance = False
        if reset_progress:
            self._step_idx = 0
            self._total_steps = None

    @staticmethod
    def _build_reference_conv_in(conv_in: nn.Conv2d) -> nn.Conv2d:
        if not isinstance(conv_in, nn.Conv2d):
            raise ValueError("UNet does not expose a Conv2d `conv_in`; cannot build 4-channel reference writer.")
        if int(conv_in.in_channels) < 4:
            raise ValueError(f"UNet conv_in expects {conv_in.in_channels} channels, cannot derive a 4-channel writer.")

        ref_conv = nn.Conv2d(
            in_channels=4,
            out_channels=conv_in.out_channels,
            kernel_size=conv_in.kernel_size,
            stride=conv_in.stride,
            padding=conv_in.padding,
            dilation=conv_in.dilation,
            groups=conv_in.groups,
            bias=conv_in.bias is not None,
            padding_mode=conv_in.padding_mode,
        )
        ref_conv = ref_conv.to(device=conv_in.weight.device, dtype=conv_in.weight.dtype)

        with torch.no_grad():
            ref_conv.weight.copy_(conv_in.weight[:, :4].contiguous())
            if conv_in.bias is not None and ref_conv.bias is not None:
                ref_conv.bias.copy_(conv_in.bias)

        return ref_conv

    @staticmethod
    def _attention_layout_signature(unet: nn.Module):
        procs = getattr(unet, "attn_processors", None)
        if procs is None:
            raise ValueError("UNet does not expose `attn_processors`; cannot verify reference alignment.")

        config = getattr(unet, "config", None)
        block_out_channels = tuple(getattr(config, "block_out_channels", ()))
        cross_attention_dim = getattr(config, "cross_attention_dim", None)

        signature = []
        for name, proc in procs.items():
            base_proc = proc.chained_proc if isinstance(proc, ReferenceOnlyAttnProc) else proc
            signature.append(
                (
                    str(name),
                    bool(getattr(proc, "enabled", False)),
                    type(proc).__name__,
                    type(base_proc).__name__,
                )
            )
        return (block_out_channels, cross_attention_dim, tuple(signature))

    def _build_reference_writer(self, reader_unet: nn.Module) -> nn.Module:
        writer_unet = copy.deepcopy(reader_unet)
        base_conv = getattr(reader_unet, "conv_in", None)
        if not isinstance(base_conv, nn.Conv2d):
            raise ValueError("UNet does not expose a Conv2d `conv_in`; cannot build reference writer.")

        writer_unet.conv_in = self._build_reference_conv_in(base_conv)
        if hasattr(writer_unet, "register_to_config"):
            try:
                writer_unet.register_to_config(in_channels=4)
            except Exception:
                pass
        writer_unet.train(reader_unet.training)
        return writer_unet

    def ensure_writer_alignment(self, *, force: bool = False):
        reader_signature = self._attention_layout_signature(self.unet)
        writer_signature = self._attention_layout_signature(self.writer_unet)
        writer_conv = getattr(self.writer_unet, "conv_in", None)

        if (
            force
            or writer_signature != reader_signature
            or not isinstance(writer_conv, nn.Conv2d)
            or int(writer_conv.in_channels) != 4
        ):
            self.writer_unet = self._build_reference_writer(self.unet)
            writer_signature = self._attention_layout_signature(self.writer_unet)

        if writer_signature != reader_signature:
            reader_keys = list(getattr(self.unet, "attn_processors", {}).keys())
            writer_keys = list(getattr(self.writer_unet, "attn_processors", {}).keys())
            raise RuntimeError(
                "Reference writer/read attention layout mismatch. "
                f"reader_layers={len(reader_keys)} writer_layers={len(writer_keys)}"
            )

    def _add_noise_like_scheduler(self, x: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
        # Most diffusers schedulers implement add_noise; fall back to x if unavailable.
        if hasattr(self.scheduler, "add_noise"):
            try:
                return self.scheduler.add_noise(x, noise, t)
            except Exception:
                return self.scheduler.add_noise(x, noise, t.reshape(-1))
        return x

    @staticmethod
    def _match_batch(x: torch.Tensor, target_b: int, *, is_cfg_guidance: bool = False) -> torch.Tensor:
        if x.shape[0] == target_b:
            return x
        if is_cfg_guidance and target_b % 2 == 0:
            half_target = target_b // 2
            if x.shape[0] == 2:
                neg = x[:1].repeat(half_target, 1, 1, 1)
                pos = x[1:2].repeat(half_target, 1, 1, 1)
                return torch.cat([neg, pos], dim=0)
            if x.shape[0] == 1:
                return x.repeat(target_b, 1, 1, 1)
        if x.shape[0] == 1:
            return x.repeat(target_b, 1, 1, 1)
        if target_b % x.shape[0] == 0:
            return x.repeat(target_b // x.shape[0], 1, 1, 1)
        if x.shape[0] > target_b:
            return x[:target_b]
        rep = [x[-1:].clone() for _ in range(target_b - x.shape[0])]
        return torch.cat([x] + rep, dim=0)

    @staticmethod
    def _prepare_write_encoder_hidden_states(
        encoder_hidden_states: torch.Tensor,
        target_b: int,
        *,
        is_cfg_guidance: bool,
    ) -> torch.Tensor:
        if not isinstance(encoder_hidden_states, torch.Tensor) or encoder_hidden_states.ndim < 3:
            return encoder_hidden_states

        ehs = encoder_hidden_states
        if is_cfg_guidance and ehs.shape[0] >= 2:
            half = ehs.shape[0] // 2
            if half > 0 and ehs.shape[0] % 2 == 0:
                # MeshGen 的 write pass 实际更接近“只用 conditional 文本上下文”。
                # 为避免 writer 同时看到 negative + positive 两路文本，把后半段
                # conditional prompt 取出后，再对齐到 writer 的 batch。
                ehs = ehs[half:]

        if ehs.shape[0] != target_b:
            if ehs.shape[0] == 1:
                ehs = ehs.repeat(target_b, 1, 1)
            elif target_b % ehs.shape[0] == 0:
                ehs = ehs.repeat(target_b // ehs.shape[0], 1, 1)
            else:
                ehs = RefOnlyNoisedUNet._match_batch(
                    ehs, target_b, is_cfg_guidance=False
                )

        return ehs

    def forward_cond(
        self,
        noisy_cond_lat,
        timestep,
        encoder_hidden_states,
        ref_dict,
        is_cfg_guidance,
        *args,
        **kwargs,
    ):
        """
        MeshGen-style write pass: run UNet on reference latent to cache K/V.
        Note: cross_attention_kwargs should already be set in kwargs by the caller (forward method).
        """
        self.ensure_writer_alignment()
        encoder_hidden_states = self._prepare_write_encoder_hidden_states(
            encoder_hidden_states,
            int(noisy_cond_lat.shape[0]) if isinstance(noisy_cond_lat, torch.Tensor) and noisy_cond_lat.ndim == 4 else 1,
            is_cfg_guidance=bool(is_cfg_guidance),
        )
        # Ensure cross_attention_kwargs is properly set (should already be set by forward)
        # CRITICAL: Use the same ref_dict object that was passed in
        if "cross_attention_kwargs" not in kwargs or not isinstance(kwargs.get("cross_attention_kwargs"), dict):
            kwargs["cross_attention_kwargs"] = {
                "mode": "w",
                "ref_dict": ref_dict,  # Use the same ref_dict object
            }
        else:
            # Update to ensure correct values, but keep the same ref_dict reference
            cak = dict(kwargs["cross_attention_kwargs"])
            cak["mode"] = "w"
            cak["ref_dict"] = ref_dict  # Ensure we use the same ref_dict object
            cak.pop("is_cfg_guidance", None)
            kwargs["cross_attention_kwargs"] = cak

        return self.writer_unet(
            noisy_cond_lat,
            timestep,
            encoder_hidden_states,
            *args,
            **kwargs,
        )

    def forward(self, sample, timestep, encoder_hidden_states, *args, **kwargs):
        cross_attention_kwargs = kwargs.get("cross_attention_kwargs", None)
        if cross_attention_kwargs is None or not isinstance(cross_attention_kwargs, dict):
            cross_attention_kwargs = {}
        cond_lat = cross_attention_kwargs.get("cond_lat")
        if cond_lat is None:
            cond_lat = self._cond_lat
        if cond_lat is None:
            # Reference not set -> behave as identity wrapper
            kwargs["cross_attention_kwargs"] = cross_attention_kwargs
            return self.unet(sample, timestep, encoder_hidden_states, *args, **kwargs)

        is_cfg_guidance = self._is_cfg_guidance
        if isinstance(cross_attention_kwargs, dict):
            is_cfg_guidance = bool(cross_attention_kwargs.get("is_cfg_guidance", is_cfg_guidance))
        processor_cfg_guidance = bool(is_cfg_guidance and _should_split_cfg_reference())

        # progress bookkeeping (per diffusion timestep)
        self._step_idx += 1
        if self._total_steps is None:
            try:
                self._total_steps = len(getattr(self.scheduler, "timesteps", [])) or None
            except Exception:
                self._total_steps = None

        # Make a fresh dict each timestep so cached keys/values don't leak across steps
        ref_dict: Dict[str, torch.Tensor] = {}

        # Debug: only print for first few steps to avoid spam
        debug_print = False
        if isinstance(timestep, torch.Tensor):
            ts_val = timestep.item() if timestep.numel() == 1 else timestep[0].item()
            # Print only for first 3 steps (assuming timesteps are in descending order, e.g., 999, 998, 997...)
            debug_print = ts_val >= 997  # Adjust threshold based on your scheduler
        elif isinstance(timestep, (int, float)):
            debug_print = timestep >= 997

        if debug_print:
            print(f"[REF] before write: len(ref_dict)={len(ref_dict)}")
        # optional terminal progress bar for write pass
        if self._enable_progress and self._total_steps:
            bar_len = 20
            cur = min(self._step_idx, self._total_steps)
            filled = int(bar_len * cur / self._total_steps)
            bar = "█" * filled + "-" * (bar_len - filled)
            print(f"[RefAttn-Write] step {cur}/{self._total_steps} {bar}")

        # Align reference batch with current call batch (important under CFG where UNet batch is 2*B).
        if isinstance(sample, torch.Tensor) and sample.ndim == 4 and isinstance(cond_lat, torch.Tensor) and cond_lat.ndim == 4:
            target_b = int(sample.shape[0])
            if cond_lat.shape[0] != target_b:
                cond_lat = self._match_batch(cond_lat, target_b, is_cfg_guidance=is_cfg_guidance)

        # Match timestep noise level: add noise to reference latent at current timestep
        noise = torch.randn_like(cond_lat)
        noisy_cond_lat = self._add_noise_like_scheduler(cond_lat, noise, timestep)
        if hasattr(self.scheduler, "scale_model_input"):
            try:
                noisy_cond_lat = self.scheduler.scale_model_input(noisy_cond_lat, timestep)
            except Exception:
                noisy_cond_lat = self.scheduler.scale_model_input(noisy_cond_lat, timestep.reshape(-1))

        # The write path always uses the explicit 4-channel writer UNet. The
        # read path still consumes the original 9-channel inpaint sample
        # unchanged, including the last 5 inpaint-specific channels.
        write_input = noisy_cond_lat

        # 1) write (cache) reference self-attn states
        #    IMPORTANT: do NOT inject ControlNet residuals during the write pass.
        #    This mirrors MeshGen behavior and avoids leaking view-specific conditions
        #    into the reference cache.
        write_kwargs = dict(kwargs)
        write_kwargs.pop("down_block_additional_residuals", None)
        write_kwargs.pop("mid_block_additional_residual", None)
        # pass mode/ref_dict/is_cfg_guidance via cross_attention_kwargs (MeshGen-style)
        # cross_attention_kwargs is already guaranteed to be a dict (normalized above)
        # CRITICAL: ref_dict must be the same object reference in both write and read passes
        cak = dict(cross_attention_kwargs)
        cak.pop("cond_lat", None)
        write_kwargs["cross_attention_kwargs"] = {
            **cak,
            "mode": "w",
            "ref_dict": ref_dict,  # Pass the same ref_dict object
            "is_cfg_guidance": processor_cfg_guidance,
        }
        if debug_print:
            print(f"[REF] write cak ref_id = {id(write_kwargs['cross_attention_kwargs']['ref_dict'])}")
        with ref_attention_context(mode="w", ref_dict=ref_dict, is_cfg_guidance=processor_cfg_guidance):
            _ = self.forward_cond(
                write_input,
                timestep,
                encoder_hidden_states,
                ref_dict,
                is_cfg_guidance,
                *args,
                **write_kwargs,
            )

        if debug_print:
            keys_sample = list(ref_dict.keys())[:3] if len(ref_dict) > 0 else []
            print(f"[REF] after write: len(ref_dict)={len(ref_dict)}  keys_sample={keys_sample}")

        # 2) read (consume) cached reference states during denoising
        # cross_attention_kwargs is already guaranteed to be a dict (normalized above)
        # CRITICAL: ref_dict must be the same object reference as in write pass
        cak = dict(cross_attention_kwargs)
        cak.pop("cond_lat", None)
        kwargs["cross_attention_kwargs"] = {
            **cak,
            "mode": "r",
            "ref_dict": ref_dict,  # Pass the same ref_dict object (should contain cached K/V from write pass)
            "is_cfg_guidance": processor_cfg_guidance,
        }
        if debug_print:
            print(f"[REF] read  cak ref_id = {id(kwargs['cross_attention_kwargs']['ref_dict'])}")
        with ref_attention_context(mode="r", ref_dict=ref_dict, is_cfg_guidance=processor_cfg_guidance):
            out = self.unet(
                sample,
                timestep,
                encoder_hidden_states,
                *args,
                **kwargs,
            )
            if debug_print:
                print(f"[REF] after read: len(ref_dict)={len(ref_dict)}")
            # optional terminal progress bar for read pass (separate from write)
            if self._enable_progress and self._total_steps:
                bar_len = 20
                cur = min(self._step_idx, self._total_steps)
                filled = int(bar_len * cur / self._total_steps)
                bar = "█" * filled + "-" * (bar_len - filled)
                print(f"[RefAttn-Read ] step {cur}/{self._total_steps} {bar}")
            return out


def ensure_ref_unet_wrapped(pipe) -> None:
    """
    Enable reference attention on `pipe.unet` and wrap it with RefOnlyNoisedUNet if not already.
    """
    if not hasattr(pipe, "unet"):
        raise ValueError("Pipeline has no `unet` attribute; cannot enable reference attention.")
    if not hasattr(pipe, "scheduler"):
        raise ValueError("Pipeline has no `scheduler` attribute; cannot enable reference attention.")

    # If already wrapped, just ensure processors are patched.
    if isinstance(pipe.unet, RefOnlyNoisedUNet):
        enable_reference_attention(pipe.unet.unet)
        pipe.unet.ensure_writer_alignment()
        return

    enable_reference_attention(pipe.unet)
    pipe.unet = RefOnlyNoisedUNet(pipe.unet, pipe.scheduler)
    pipe.unet.ensure_writer_alignment()


def debug_refattn(pipe) -> None:
    """
    Minimal runtime prints to verify:
    - pipe.unet is wrapped by RefOnlyNoisedUNet
    - base_unet.attn_processors contains ReferenceOnlyAttnProc wrappers
    - how many wrappers are enabled (typically attn1 self-attn only)
    """
    try:
        unet_obj = getattr(pipe, "unet", None)
        print("[DBG][RefAttn] pipe.unet type =", type(unet_obj))
        print("[DBG][RefAttn] pipe.unet class name =", type(unet_obj).__name__)
        print("[DBG][RefAttn] pipe.unet has .unet =", hasattr(unet_obj, "unet"))

        base_unet = getattr(unet_obj, "unet", unet_obj)
        print("[DBG][RefAttn] base_unet type =", type(base_unet))

        procs = getattr(base_unet, "attn_processors", None)
        if procs is None:
            print("[DBG][RefAttn] base_unet has no attn_processors -> cannot verify processors")
            return

        proc_types = {}
        wrapped = 0
        enabled = 0
        wrapped_samples = []
        enabled_key_samples = []

        for k, v in procs.items():
            tname = type(v).__name__
            proc_types[tname] = proc_types.get(tname, 0) + 1
            if tname == "ReferenceOnlyAttnProc":
                wrapped += 1
                en = bool(getattr(v, "enabled", False))
                if en:
                    enabled += 1
                    if len(enabled_key_samples) < 5:
                        enabled_key_samples.append(k)
                if len(wrapped_samples) < 3:
                    wrapped_samples.append((k, en))

        print("[DBG][RefAttn] attn_processors total =", len(procs))
        print("[DBG][RefAttn] type histogram =", proc_types)
        print("[DBG][RefAttn] ReferenceOnlyAttnProc wrapped =", wrapped)
        print("[DBG][RefAttn] ReferenceOnlyAttnProc enabled =", enabled)
        print("[DBG][RefAttn] wrapped samples =", wrapped_samples)
        print("[DBG][RefAttn] enabled key samples =", enabled_key_samples)
    except Exception as e:
        print("[DBG][RefAttn] debug_refattn failed:", repr(e))
