"""
Reference Attention (minimal port inspired by MeshGen/zero123pp)

Goal:
- During denoising, inject K/V from the reference image's *self-attention*
  into the current denoising step, to improve consistency with the reference image.

Implementation approach (diffusers-friendly, no pipeline changes required):
- Wrap UNet's self-attention processors (attn1) with a processor that can:
  - mode="w": write (cache) encoder_hidden_states
    (for self-attn: hidden_states) into a ref_dict
  - mode="r": read cached states and concatenate on the sequence dimension
- Wrap UNet forward:
  - For each timestep, first run UNet on a *noised reference latent*
    in write mode to populate ref_dict
  - Then run UNet on current sample latent in read mode to consume ref_dict

Note:
- We avoid relying on `cross_attention_kwargs` since some bundled diffusers
  versions don't forward it.
- Instead, we use a small module-level context
  (mode/ref_dict/is_cfg_guidance) set by the UNet wrapper.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class _RefAttnState:
    mode: str = "off"  # "w" | "r" | "m" | "off"
    ref_dict: Optional[Dict[str, torch.Tensor]] = None
    is_cfg_guidance: bool = False


_STATE = _RefAttnState()


@contextmanager
def ref_attention_context(
    *,
    mode: str,
    ref_dict: Dict[str, torch.Tensor],
    is_cfg_guidance: bool,
):
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
    A wrapper for diffusers attention processors
    (e.g., AttnProcessor2_0, XFormersAttnProcessor, etc.).

    It only activates when enabled=True and global ref_attention_context
    is set (mode != "off").
    """

    def __init__(
        self,
        chained_proc: Any,
        enabled: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.chained_proc = chained_proc
        self.name = name or "attn"

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        **kwargs,
    ):
        # Keep behavior identical when disabled
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # Extract mode/ref_dict/is_cfg_guidance from kwargs
        # (diffusers expands cross_attention_kwargs into kwargs)
        # Fallback to global context if not in kwargs (for backward compatibility)
        # CRITICAL: Use the module-level _STATE directly (same file),
        # don't re-import to avoid module duplication
        if (
            "mode" in kwargs
            and "ref_dict" in kwargs
            and "is_cfg_guidance" in kwargs
        ):
            mode = kwargs.pop("mode")
            ref_dict = kwargs.pop("ref_dict")
            is_cfg_guidance = kwargs.pop("is_cfg_guidance")
            source = "kwargs"
        else:
            # Fallback to global context
            # (should not happen if cross_attention_kwargs is properly set)
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
        if hasattr(self, "_debug_step_count"):
            self._debug_step_count = getattr(self, "_debug_step_count", 0) + 1
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

        # 1: 进入 write/read 前，只对 attn1（self.enabled）打一次
        if debug_attn and self.enabled:
            hs_sh = tuple(hidden_states.shape)
            ehs_sh = tuple(encoder_hidden_states.shape)
            print(
                f"[ATTN] {self.name} ENABLED mode={mode} src={source} "
                f"hs={hs_sh} ehs={ehs_sh} "
                f"dict_len={len(ref_dict)} dict_id={id(ref_dict)}"
            )

        # CFG 污染检查：batch 与 is_cfg 是否匹配
        if debug_attn and self.enabled and mode in ("w", "r"):
            print(
                f"[CFGCHK] {self.name} "
                f"batch={hidden_states.shape[0]} is_cfg={is_cfg_guidance}"
            )

        # In CFG, UNet is called with a batch of size 2*B laid out as
        # [uncond..., cond...]. Keep the unconditional half isolated so
        # reference write/read only affects the conditional half.
        if is_cfg_guidance and self.enabled:
            cfg_batch = (
                hidden_states.shape[0] // 2
                if hidden_states.shape[0] % 2 == 0
                else 1
            )
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
                    print(
                        f"[W] {self.name} stored "
                        f"shape={tuple(encoder_hidden_states.shape)} "
                        f"dict_len={len(ref_dict)}"
                    )
            elif mode == "r":
                if self.name not in ref_dict:
                    print(
                        f"[R][WARN] {self.name} missing in ref_dict! "
                        f"dict_keys_sample={list(ref_dict.keys())[:5]}"
                    )
                else:
                    before = tuple(encoder_hidden_states.shape)
                    ref = ref_dict.pop(self.name)
                    encoder_hidden_states = torch.cat(
                        [encoder_hidden_states, ref], dim=1
                    )
                    after = tuple(encoder_hidden_states.shape)
                    if debug_attn:
                        print(
                            f"[R] {self.name} cat {before} + {tuple(ref.shape)} "
                            f"-> {after} dict_len={len(ref_dict)}"
                        )
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
    if getattr(unet, "_easitex_ref_attn_enabled", False):
        return unet

    if not hasattr(unet, "attn_processors"):
        raise ValueError(
            "UNet does not expose attn_processors; "
            "cannot enable reference attention."
        )
    if not hasattr(unet, "set_attn_processor"):
        raise ValueError(
            "UNet does not expose set_attn_processor; "
            "cannot enable reference attention."
        )

    new_procs = {}
    for name, proc in unet.attn_processors.items():
        # Diffusers convention: attn1 = self-attention, attn2 = cross-attention
        # Some diffusers versions use keys like "...attn1.processor",
        # some may differ slightly; enabling on any "attn1" is the safest
        # to make sure reference attention actually activates.
        n = str(name)
        enabled = ("attn1" in n) and ("attn2" not in n)
        new_procs[name] = ReferenceOnlyAttnProc(
            proc,
            enabled=enabled,
            name=name,
        )

    unet.set_attn_processor(new_procs)
    setattr(unet, "_easitex_ref_attn_enabled", True)
    return unet


class RefOnlyNoisedUNet(nn.Module):
    """
    A UNet wrapper that, for each forward call (timestep), performs:
      1) run UNet on a *noised reference latent* in write mode to populate ref_dict
      2) run UNet on the real sample latent in read mode to consume ref_dict

    The reference latent (cond_lat) must be set via set_cond_lat(...) before sampling.
    """

    def __init__(self, unet: nn.Module, scheduler):
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler
        self._cond_lat: Optional[torch.Tensor] = None

        # For inpainting UNets (in_channels=9), we may need the extra 5 channels:
        # mask(1) + masked_latent(4)
        # Some pipelines pass the packed 9ch sample, but for robustness we also allow
        # callers to set it explicitly via set_inpaint_extra.
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

    def set_cond_lat(self, cond_lat: torch.Tensor, *, is_cfg_guidance: bool):
        self._cond_lat = cond_lat
        self._is_cfg_guidance = bool(is_cfg_guidance)

    def set_inpaint_extra(self, cond_extra: Optional[torch.Tensor]):
        """
        Set extra channels for inpainting UNet input:
        - cond_extra should be (B, 5, h, w) corresponding to mask(1) + masked_latent(4)
        - If None, wrapper falls back to zeros for the write pass.
        """
        self._cond_extra = cond_extra

    def clear_reference_conditioning(self, *, reset_progress: bool = True):
        self._cond_lat = None
        self._cond_extra = None
        self._is_cfg_guidance = False

        if reset_progress:
            self._step_idx = 0
            self._total_steps = None

    def _add_noise_like_scheduler(
        self,
        x: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ):
        # Most diffusers schedulers implement add_noise; fall back to x if unavailable.
        if hasattr(self.scheduler, "add_noise"):
            try:
                return self.scheduler.add_noise(x, noise, t)
            except Exception:
                return self.scheduler.add_noise(x, noise, t.reshape(-1))
        return x

    @staticmethod
    def _match_batch(
        x: torch.Tensor,
        target_b: int,
        *,
        is_cfg_guidance: bool = False,
    ) -> torch.Tensor:
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

        Note:
            cross_attention_kwargs should already be set in kwargs by the caller
            (forward method).
        """
        # Ensure cross_attention_kwargs is properly set
        # (should already be set by forward)
        # CRITICAL: Use the same ref_dict object that was passed in
        if (
            "cross_attention_kwargs" not in kwargs
            or not isinstance(kwargs.get("cross_attention_kwargs"), dict)
        ):
            kwargs["cross_attention_kwargs"] = {
                "mode": "w",
                "ref_dict": ref_dict,  # Use the same ref_dict object
                "is_cfg_guidance": bool(is_cfg_guidance),
            }
        else:
            # Update to ensure correct values, but keep the same ref_dict reference
            cak = dict(kwargs["cross_attention_kwargs"])
            cak["mode"] = "w"
            cak["ref_dict"] = ref_dict
            cak["is_cfg_guidance"] = bool(is_cfg_guidance)
            kwargs["cross_attention_kwargs"] = cak

        return self.unet(
            noisy_cond_lat,
            timestep,
            encoder_hidden_states,
            *args,
            **kwargs,
        )

    def forward(self, sample, timestep, encoder_hidden_states, *args, **kwargs):
        cross_attention_kwargs = kwargs.get("cross_attention_kwargs", None)
        if cross_attention_kwargs is None or not isinstance(
            cross_attention_kwargs, dict
        ):
            cross_attention_kwargs = {}

        cond_lat = cross_attention_kwargs.get("cond_lat")
        if cond_lat is None:
            cond_lat = self._cond_lat

        if cond_lat is None:
            # Reference not set -> behave as identity wrapper
            kwargs["cross_attention_kwargs"] = cross_attention_kwargs
            return self.unet(
                sample,
                timestep,
                encoder_hidden_states,
                *args,
                **kwargs,
            )

        is_cfg_guidance = self._is_cfg_guidance
        if isinstance(cross_attention_kwargs, dict):
            is_cfg_guidance = bool(
                cross_attention_kwargs.get(
                    "is_cfg_guidance",
                    is_cfg_guidance,
                )
            )

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
            # Print only for first 3 steps (assuming timesteps are in descending order)
            debug_print = ts_val >= 997
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

        # Align reference batch with current call batch
        # (important under CFG where UNet batch is 2*B)
        if (
            isinstance(sample, torch.Tensor)
            and sample.ndim == 4
            and isinstance(cond_lat, torch.Tensor)
            and cond_lat.ndim == 4
        ):
            target_b = int(sample.shape[0])
            if cond_lat.shape[0] != target_b:
                cond_lat = self._match_batch(
                    cond_lat,
                    target_b,
                    is_cfg_guidance=is_cfg_guidance,
                )

        # Match timestep noise level: add noise to reference latent at current timestep
        noise = torch.randn_like(cond_lat)
        noisy_cond_lat = self._add_noise_like_scheduler(cond_lat, noise, timestep)

        if hasattr(self.scheduler, "scale_model_input"):
            try:
                noisy_cond_lat = self.scheduler.scale_model_input(
                    noisy_cond_lat,
                    timestep,
                )
            except Exception:
                noisy_cond_lat = self.scheduler.scale_model_input(
                    noisy_cond_lat,
                    timestep.reshape(-1),
                )

        # Ensure write-pass input channels match underlying UNet expected in_channels
        # (e.g. 9 for inpaint).
        expected_c = None
        try:
            expected_c = int(
                getattr(getattr(self.unet, "config", None), "in_channels", None)
            )
        except Exception:
            expected_c = None

        write_input = noisy_cond_lat
        if (
            expected_c is not None
            and isinstance(write_input, torch.Tensor)
            and write_input.ndim == 4
        ):
            cur_c = int(write_input.shape[1])

            if cur_c < expected_c:
                need = expected_c - cur_c
                extra = None

                # Use explicit reference-side inpaint channels when available.
                # Do not reuse the current sample's packed 9-channel input here:
                # those extra channels belong to the current inpaint view and would
                # leak view-specific mask/masked-image information into the reference cache.
                if isinstance(self._cond_extra, torch.Tensor) and self._cond_extra.ndim == 4:
                    extra = self._match_batch(
                        self._cond_extra,
                        int(write_input.shape[0]),
                        is_cfg_guidance=is_cfg_guidance,
                    )
                    if (
                        extra.shape[2:] != write_input.shape[2:]
                        or int(extra.shape[1]) != need
                    ):
                        extra = None

                # Last resort: pad zeros to keep the write pass reference-only.
                if extra is None:
                    extra = torch.zeros(
                        (
                            int(write_input.shape[0]),
                            need,
                            int(write_input.shape[2]),
                            int(write_input.shape[3]),
                        ),
                        device=write_input.device,
                        dtype=write_input.dtype,
                    )

                write_input = torch.cat([write_input, extra], dim=1)

            elif cur_c > expected_c:
                write_input = write_input[:, :expected_c]

        # 1) write (cache) reference self-attn states
        # IMPORTANT: do NOT inject ControlNet residuals during the write pass.
        # This mirrors MeshGen behavior and avoids leaking view-specific conditions
        # into the reference cache.
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
            "ref_dict": ref_dict,
            "is_cfg_guidance": is_cfg_guidance,
        }

        if debug_print:
            print(
                "[REF] write cak ref_id = "
                f"{id(write_kwargs['cross_attention_kwargs']['ref_dict'])}"
            )

        with ref_attention_context(
            mode="w",
            ref_dict=ref_dict,
            is_cfg_guidance=is_cfg_guidance,
        ):
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
            print(
                f"[REF] after write: len(ref_dict)={len(ref_dict)} "
                f"keys_sample={keys_sample}"
            )

        # 2) read (consume) cached reference states during denoising
        # cross_attention_kwargs is already guaranteed to be a dict (normalized above)
        # CRITICAL: ref_dict must be the same object reference as in write pass
        cak = dict(cross_attention_kwargs)
        cak.pop("cond_lat", None)

        kwargs["cross_attention_kwargs"] = {
            **cak,
            "mode": "r",
            "ref_dict": ref_dict,
            "is_cfg_guidance": is_cfg_guidance,
        }

        if debug_print:
            print(
                "[REF] read cak ref_id = "
                f"{id(kwargs['cross_attention_kwargs']['ref_dict'])}"
            )

        with ref_attention_context(
            mode="r",
            ref_dict=ref_dict,
            is_cfg_guidance=is_cfg_guidance,
        ):
            out = self.unet(
                sample,
                timestep,
                encoder_hidden_states,
                *args,
                **kwargs,
            )

            if debug_print:
                print(f"[REF] after read: len(ref_dict)={len(ref_dict)}")

            # optional terminal progress bar for read pass
            if self._enable_progress and self._total_steps:
                bar_len = 20
                cur = min(self._step_idx, self._total_steps)
                filled = int(bar_len * cur / self._total_steps)
                bar = "█" * filled + "-" * (bar_len - filled)
                print(f"[RefAttn-Read ] step {cur}/{self._total_steps} {bar}")

            return out


def ensure_ref_unet_wrapped(pipe) -> None:
    """
    Enable reference attention on pipe.unet and wrap it with RefOnlyNoisedUNet
    if not already.
    """
    if not hasattr(pipe, "unet"):
        raise ValueError(
            "Pipeline has no unet attribute; cannot enable reference attention."
        )
    if not hasattr(pipe, "scheduler"):
        raise ValueError(
            "Pipeline has no scheduler attribute; cannot enable reference attention."
        )

    # If already wrapped, just ensure processors are patched.
    if isinstance(pipe.unet, RefOnlyNoisedUNet):
        enable_reference_attention(pipe.unet.unet)
        return

    enable_reference_attention(pipe.unet)
    pipe.unet = RefOnlyNoisedUNet(pipe.unet, pipe.scheduler)


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
            print(
                "[DBG][RefAttn] base_unet has no attn_processors "
                "-> cannot verify processors"
            )
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