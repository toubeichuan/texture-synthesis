"""
4-channel reference attention with MeshGen-style single-UNet read/write passes.

Design goals:
- Only patch attention processors and add one wrapper outside the UNet.
- Only affect self-attention (`attn1`), never touch cross-attention (`attn2`).
- Build a fresh `ref_dict` on every timestep.
- Run the write pass on the same UNet at the same timestep noise level.
- Keep ControlNet residuals out of the reference write pass.
- Under CFG, keep the unconditional branch on the original attention path.
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
    mode: str = "off"
    ref_dict: Optional[Dict[str, torch.Tensor]] = None
    is_cfg_guidance: bool = False
    cfg_layout: Optional[str] = None
    runtime_checks: bool = False
    processor_kwargs_reported: bool = False


_STATE = _RefAttnState()


def _runtime_checks_enabled() -> bool:
    return os.environ.get("EASITEX_REF_ATTN_RUNTIME_CHECKS", "0") != "0"


def _runtime_checks_max_steps() -> int:
    raw = os.environ.get("EASITEX_REF_ATTN_RUNTIME_CHECKS_MAX_STEPS", "6")
    try:
        return max(int(raw), 1)
    except Exception:
        return 6


def _runtime_log(message: str, *, step_idx: Optional[int] = None, level: str = "INFO", force: bool = False) -> None:
    if not _runtime_checks_enabled():
        return
    if not force and step_idx is not None and step_idx > _runtime_checks_max_steps():
        return
    prefix = f"[RefAttnCheck][{level}]"
    if step_idx is not None:
        prefix += f"[step={step_idx}]"
    print(f"{prefix} {message}")


def _tensor_storage_ptr(x: torch.Tensor) -> int:
    try:
        return int(x.untyped_storage().data_ptr())
    except Exception:
        return int(x.data_ptr())


def _format_timestep(timestep) -> str:
    if isinstance(timestep, torch.Tensor):
        if timestep.numel() == 1:
            return str(float(timestep.detach().float().reshape(-1)[0].item()))
        first = float(timestep.detach().float().reshape(-1)[0].item())
        return f"{first}x{tuple(timestep.shape)}"
    return str(timestep)


def _repeat_like_batch(x: torch.Tensor, target_batch: int) -> torch.Tensor:
    if x.shape[0] == target_batch:
        return x

    if x.shape[0] == 1:
        repeats = [target_batch] + [1] * (x.ndim - 1)
        return x.repeat(*repeats)

    if target_batch % x.shape[0] == 0:
        repeats = [target_batch // x.shape[0]] + [1] * (x.ndim - 1)
        return x.repeat(*repeats)

    if x.shape[0] > target_batch:
        return x[:target_batch]

    tail = [x[-1:].clone() for _ in range(target_batch - x.shape[0])]
    return torch.cat([x] + tail, dim=0)


def _take_cfg_conditional_half(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor) or x.ndim == 0:
        return x
    if x.shape[0] < 2 or x.shape[0] % 2 != 0:
        return x
    return x[x.shape[0] // 2 :]


def _is_cfg_batch(hidden_states: torch.Tensor, is_cfg_guidance: bool) -> bool:
    return bool(is_cfg_guidance and hidden_states.shape[0] >= 2 and hidden_states.shape[0] % 2 == 0)


def _get_cfg_slices(batch_size: int, is_cfg_guidance: bool, cfg_layout: Optional[str]):
    if not bool(is_cfg_guidance and batch_size >= 2 and batch_size % 2 == 0):
        return None

    split_idx = batch_size // 2
    if cfg_layout == "uncond_first":
        return slice(0, split_idx), slice(split_idx, batch_size)
    if cfg_layout == "cond_first":
        return slice(split_idx, batch_size), slice(0, split_idx)
    return None


def _take_cfg_branch(
    x: torch.Tensor,
    *,
    is_cfg_guidance: bool,
    cfg_layout: Optional[str],
    branch: str,
) -> torch.Tensor:
    if not isinstance(x, torch.Tensor) or x.ndim == 0:
        return x

    cfg_slices = _get_cfg_slices(int(x.shape[0]), is_cfg_guidance, cfg_layout)
    if cfg_slices is None:
        return x

    uncond_slice, cond_slice = cfg_slices
    if branch == "cond":
        return x[cond_slice]
    if branch == "uncond":
        return x[uncond_slice]
    raise ValueError(f"Unsupported CFG branch: {branch}")


@contextmanager
def ref_attention_context(
    *,
    mode: str,
    ref_dict: Dict[str, torch.Tensor],
    is_cfg_guidance: bool,
    cfg_layout: Optional[str] = None,
    runtime_checks: bool = False,
):
    prev = (
        _STATE.mode,
        _STATE.ref_dict,
        _STATE.is_cfg_guidance,
        _STATE.cfg_layout,
        _STATE.runtime_checks,
        _STATE.processor_kwargs_reported,
    )
    _STATE.mode = str(mode)
    _STATE.ref_dict = ref_dict
    _STATE.is_cfg_guidance = bool(is_cfg_guidance)
    _STATE.cfg_layout = cfg_layout
    _STATE.runtime_checks = bool(runtime_checks)
    _STATE.processor_kwargs_reported = False
    try:
        yield
    finally:
        (
            _STATE.mode,
            _STATE.ref_dict,
            _STATE.is_cfg_guidance,
            _STATE.cfg_layout,
            _STATE.runtime_checks,
            _STATE.processor_kwargs_reported,
        ) = prev


class ReferenceOnlyAttnProc(nn.Module):
    def __init__(self, chained_proc: Any, enabled: bool = False, name: Optional[str] = None):
        super().__init__()
        self.enabled = bool(enabled)
        self.chained_proc = chained_proc
        self.name = name or "attn"

    def _call_base(self, attn, hidden_states, encoder_hidden_states, attention_mask, temb, kwargs):
        return self.chained_proc(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            **kwargs,
        )

    def _apply_reference(self, mode: str, encoder_hidden_states: torch.Tensor, ref_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        if mode == "w":
            ref_dict[self.name] = encoder_hidden_states
            return encoder_hidden_states

        if mode != "r":
            return encoder_hidden_states

        ref_states = ref_dict.pop(self.name, None)
        if ref_states is None:
            return encoder_hidden_states

        if ref_states.shape[0] != encoder_hidden_states.shape[0]:
            ref_states = _repeat_like_batch(ref_states, encoder_hidden_states.shape[0])

        return torch.cat([encoder_hidden_states, ref_states], dim=1)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        has_direct_ref_kwargs = any(
            key in kwargs for key in ("mode", "ref_dict", "is_cfg_guidance", "cfg_layout")
        )
        mode = str(kwargs.pop("mode", _STATE.mode))
        ref_dict = kwargs.pop("ref_dict", _STATE.ref_dict)
        is_cfg_guidance = bool(kwargs.pop("is_cfg_guidance", _STATE.is_cfg_guidance))
        cfg_layout = kwargs.pop("cfg_layout", _STATE.cfg_layout)

        if self.enabled and _STATE.runtime_checks and not _STATE.processor_kwargs_reported:
            source = "cross_attention_kwargs" if has_direct_ref_kwargs else "context_fallback"
            level = "PASS" if has_direct_ref_kwargs else "WARN"
            _runtime_log(
                f"processor received ref-attn controls via {source} on layer `{self.name}` (mode={mode})",
                level=level,
                force=True,
            )
            _STATE.processor_kwargs_reported = True

        if not self.enabled or mode not in {"w", "r"} or ref_dict is None:
            return self._call_base(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                temb,
                kwargs,
            )

        cfg_slices = _get_cfg_slices(int(hidden_states.shape[0]), is_cfg_guidance, cfg_layout)
        if cfg_slices is not None:
            uncond_slice, cond_slice = cfg_slices

            uncond_out = self._call_base(
                attn,
                hidden_states[uncond_slice],
                encoder_hidden_states[uncond_slice],
                attention_mask,
                temb,
                kwargs,
            )

            cond_encoder_hidden_states = self._apply_reference(
                mode,
                encoder_hidden_states[cond_slice],
                ref_dict,
            )
            cond_out = self._call_base(
                attn,
                hidden_states[cond_slice],
                cond_encoder_hidden_states,
                attention_mask,
                temb,
                kwargs,
            )
            return torch.cat([uncond_out, cond_out], dim=0)

        encoder_hidden_states = self._apply_reference(mode, encoder_hidden_states, ref_dict)
        return self._call_base(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            temb,
            kwargs,
        )


def _get_unet_in_channels(unet: nn.Module) -> Optional[int]:
    conv_in = getattr(unet, "conv_in", None)
    if isinstance(conv_in, nn.Conv2d):
        return int(conv_in.in_channels)
    return None


def _require_4_channel_unet(unet: nn.Module) -> None:
    in_channels = _get_unet_in_channels(unet)
    if in_channels is None:
        return
    if in_channels != 4:
        raise ValueError(
            f"reference_attention_new_4Channel only supports 4-channel UNet inputs, got {in_channels} channels."
        )


def enable_reference_attention(unet: nn.Module) -> nn.Module:
    if not hasattr(unet, "attn_processors"):
        raise ValueError("UNet does not expose `attn_processors`; cannot enable reference attention.")
    if not hasattr(unet, "set_attn_processor"):
        raise ValueError("UNet does not expose `set_attn_processor`; cannot enable reference attention.")

    new_procs = {}
    for name, proc in unet.attn_processors.items():
        base_proc = proc.chained_proc if isinstance(proc, ReferenceOnlyAttnProc) else proc
        enabled = "attn1" in str(name) and "attn2" not in str(name)
        new_procs[name] = ReferenceOnlyAttnProc(base_proc, enabled=enabled, name=name)

    unet.set_attn_processor(new_procs)
    setattr(unet, "_easitex_ref_attn_enabled", True)
    return unet


class RefOnlyNoisedUNet(nn.Module):
    def __init__(self, unet: nn.Module, scheduler):
        super().__init__()
        _require_4_channel_unet(unet)
        self.unet = unet
        self.scheduler = scheduler
        self._cond_lat: Optional[torch.Tensor] = None
        self._is_cfg_guidance: bool = False
        self._cfg_layout: Optional[str] = None
        self._runtime_step_idx: int = 0
        self._prev_ref_dict: Optional[Dict[str, torch.Tensor]] = None
        self._warned_missing_cfg_layout: bool = False

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)

    def set_cond_lat(self, cond_lat: torch.Tensor, *, is_cfg_guidance: bool, cfg_layout: Optional[str] = None):
        if cond_lat is None:
            self._cond_lat = None
            self._is_cfg_guidance = False
            self._cfg_layout = None
            return
        if cond_lat.ndim != 4 or cond_lat.shape[1] != 4:
            raise ValueError(
                f"Reference latent must be BCHW with 4 channels, got shape {tuple(cond_lat.shape)}."
            )
        self._cond_lat = cond_lat
        self._is_cfg_guidance = bool(is_cfg_guidance)
        self._cfg_layout = cfg_layout

    def set_inpaint_extra(self, cond_extra: Optional[torch.Tensor]):
        if cond_extra is not None:
            raise ValueError("reference_attention_new_4Channel does not support 9-channel inpaint extras.")

    def clear_reference_conditioning(self, *, reset_progress: bool = True):
        if reset_progress:
            self._runtime_step_idx = 0
        self._cond_lat = None
        self._is_cfg_guidance = False
        self._cfg_layout = None
        self._prev_ref_dict = None
        self._warned_missing_cfg_layout = False

    def ensure_writer_alignment(self, *, force: bool = False):
        del force
        _require_4_channel_unet(self.unet)

    @staticmethod
    def _add_noise_like_scheduler(scheduler, latents: torch.Tensor, noise: torch.Tensor, timestep):
        if not hasattr(scheduler, "add_noise"):
            return latents

        try:
            return scheduler.add_noise(latents, noise, timestep)
        except Exception:
            if isinstance(timestep, torch.Tensor):
                return scheduler.add_noise(latents, noise, timestep.reshape(-1))
            raise

    @staticmethod
    def _scale_model_input_like_scheduler(scheduler, latents: torch.Tensor, timestep):
        if not hasattr(scheduler, "scale_model_input"):
            return latents

        try:
            return scheduler.scale_model_input(latents, timestep)
        except Exception:
            if isinstance(timestep, torch.Tensor):
                return scheduler.scale_model_input(latents, timestep.reshape(-1))
            raise

    @staticmethod
    def _prepare_write_encoder_hidden_states(
        encoder_hidden_states: torch.Tensor,
        target_batch: int,
        *,
        is_cfg_guidance: bool,
        cfg_layout: Optional[str],
    ) -> torch.Tensor:
        if not isinstance(encoder_hidden_states, torch.Tensor):
            return encoder_hidden_states

        write_states = (
            _take_cfg_branch(
                encoder_hidden_states,
                is_cfg_guidance=is_cfg_guidance,
                cfg_layout=cfg_layout,
                branch="cond",
            )
            if is_cfg_guidance
            else encoder_hidden_states
        )
        return _repeat_like_batch(write_states, target_batch)

    @staticmethod
    def _prepare_write_class_labels(
        class_labels,
        target_batch: int,
        *,
        is_cfg_guidance: bool,
        cfg_layout: Optional[str],
    ):
        if not isinstance(class_labels, torch.Tensor):
            return class_labels

        write_labels = class_labels
        if is_cfg_guidance:
            write_labels = _take_cfg_branch(
                class_labels,
                is_cfg_guidance=is_cfg_guidance,
                cfg_layout=cfg_layout,
                branch="cond",
            )
        return _repeat_like_batch(write_labels, target_batch)

    @staticmethod
    def _prepare_write_latents(
        cond_lat: torch.Tensor,
        sample_batch: int,
        *,
        is_cfg_guidance: bool,
        cfg_layout: Optional[str],
    ) -> torch.Tensor:
        cfg_slices = _get_cfg_slices(sample_batch, is_cfg_guidance, cfg_layout)
        if cfg_slices is not None:
            target_batch = sample_batch // 2
            write_latents = _take_cfg_branch(
                cond_lat,
                is_cfg_guidance=is_cfg_guidance,
                cfg_layout=cfg_layout,
                branch="cond",
            )
        else:
            target_batch = sample_batch
            write_latents = cond_lat
        return _repeat_like_batch(write_latents, target_batch)

    def forward(self, sample, timestep, encoder_hidden_states, *args, **kwargs):
        cross_attention_kwargs = kwargs.get("cross_attention_kwargs")
        if not isinstance(cross_attention_kwargs, dict):
            cross_attention_kwargs = {}
        else:
            cross_attention_kwargs = dict(cross_attention_kwargs)

        cond_lat = cross_attention_kwargs.get("cond_lat", self._cond_lat)
        if cond_lat is None:
            kwargs["cross_attention_kwargs"] = cross_attention_kwargs
            return self.unet(sample, timestep, encoder_hidden_states, *args, **kwargs)

        _require_4_channel_unet(self.unet)
        if not isinstance(sample, torch.Tensor) or sample.ndim != 4 or sample.shape[1] != 4:
            raise ValueError(
                f"reference_attention_new_4Channel expects 4-channel UNet samples, got shape {tuple(sample.shape)}."
            )

        cond_lat = cond_lat.to(device=sample.device, dtype=sample.dtype)
        is_cfg_guidance = bool(cross_attention_kwargs.get("is_cfg_guidance", self._is_cfg_guidance))
        cfg_layout = cross_attention_kwargs.get("cfg_layout", self._cfg_layout)
        runtime_checks = _runtime_checks_enabled()

        self._runtime_step_idx += 1
        step_idx = self._runtime_step_idx

        if is_cfg_guidance and cfg_layout not in {"uncond_first", "cond_first"} and not self._warned_missing_cfg_layout:
            _runtime_log(
                "CFG is enabled but `cfg_layout` was not explicitly confirmed; batch splitting will be skipped.",
                step_idx=step_idx,
                level="WARN",
                force=True,
            )
            self._warned_missing_cfg_layout = True

        ref_dict: Dict[str, torch.Tensor] = {}
        if self._prev_ref_dict is ref_dict:
            raise AssertionError("ref_dict must be rebuilt per timestep, but the same dict object was reused.")
        self._prev_ref_dict = ref_dict
        _runtime_log(
            f"rebuilt ref_dict id={id(ref_dict)} for timestep={_format_timestep(timestep)}",
            step_idx=step_idx,
        )

        write_source_latents = self._prepare_write_latents(
            cond_lat,
            int(sample.shape[0]),
            is_cfg_guidance=is_cfg_guidance,
            cfg_layout=cfg_layout,
        )

        sample_ptr = _tensor_storage_ptr(sample)
        ref_ptr = _tensor_storage_ptr(write_source_latents)
        if sample_ptr == ref_ptr:
            raise AssertionError("reference latent and current sample latent unexpectedly share storage.")
        _runtime_log(
            f"role separation confirmed: sample_ptr={sample_ptr} ref_ptr={ref_ptr} "
            f"sample_shape={tuple(sample.shape)} ref_shape={tuple(write_source_latents.shape)}",
            step_idx=step_idx,
        )

        noise = torch.randn(
            write_source_latents.shape,
            device=write_source_latents.device,
            dtype=write_source_latents.dtype,
        )
        noised_write_latents = self._add_noise_like_scheduler(
            self.scheduler,
            write_source_latents,
            noise,
            timestep,
        )
        noise_delta = float((noised_write_latents - write_source_latents).abs().mean().item())
        _runtime_log(
            f"re-noised reference latent at timestep={_format_timestep(timestep)} "
            f"(mean_abs_delta={noise_delta:.6f})",
            step_idx=step_idx,
        )
        write_latents = noised_write_latents
        write_latents = self._scale_model_input_like_scheduler(self.scheduler, write_latents, timestep)

        write_kwargs = dict(kwargs)
        incoming_down = write_kwargs.get("down_block_additional_residuals") is not None
        incoming_mid = write_kwargs.get("mid_block_additional_residual") is not None
        write_kwargs.pop("down_block_additional_residuals", None)
        write_kwargs.pop("mid_block_additional_residual", None)
        write_has_down = write_kwargs.get("down_block_additional_residuals") is not None
        write_has_mid = write_kwargs.get("mid_block_additional_residual") is not None
        if write_has_down or write_has_mid:
            raise AssertionError("ControlNet residuals leaked into the reference write pass.")
        _runtime_log(
            f"ControlNet residual routing confirmed: incoming_down={incoming_down} incoming_mid={incoming_mid} "
            f"write_down={write_has_down} write_mid={write_has_mid}",
            step_idx=step_idx,
        )

        class_labels = write_kwargs.get("class_labels")
        if class_labels is not None:
            write_kwargs["class_labels"] = self._prepare_write_class_labels(
                class_labels,
                int(write_latents.shape[0]),
                is_cfg_guidance=is_cfg_guidance,
                cfg_layout=cfg_layout,
            )

        write_encoder_hidden_states = self._prepare_write_encoder_hidden_states(
            encoder_hidden_states,
            int(write_latents.shape[0]),
            is_cfg_guidance=is_cfg_guidance,
            cfg_layout=cfg_layout,
        )

        write_cross_attention_kwargs = dict(cross_attention_kwargs)
        write_cross_attention_kwargs.pop("cond_lat", None)
        write_cross_attention_kwargs["mode"] = "w"
        write_cross_attention_kwargs["ref_dict"] = ref_dict
        write_cross_attention_kwargs["is_cfg_guidance"] = False
        write_cross_attention_kwargs["cfg_layout"] = cfg_layout
        write_kwargs["cross_attention_kwargs"] = write_cross_attention_kwargs

        with ref_attention_context(
            mode="w",
            ref_dict=ref_dict,
            is_cfg_guidance=False,
            cfg_layout=cfg_layout,
            runtime_checks=runtime_checks,
        ):
            self.unet(
                write_latents,
                timestep,
                write_encoder_hidden_states,
                *args,
                **write_kwargs,
            )

        read_cross_attention_kwargs = dict(cross_attention_kwargs)
        read_cross_attention_kwargs.pop("cond_lat", None)
        read_cross_attention_kwargs["mode"] = "r"
        read_cross_attention_kwargs["ref_dict"] = ref_dict
        read_cross_attention_kwargs["is_cfg_guidance"] = is_cfg_guidance
        read_cross_attention_kwargs["cfg_layout"] = cfg_layout
        kwargs["cross_attention_kwargs"] = read_cross_attention_kwargs

        _runtime_log(
            f"read pass uses real sample latents with ControlNet residuals: "
            f"down={kwargs.get('down_block_additional_residuals') is not None} "
            f"mid={kwargs.get('mid_block_additional_residual') is not None}",
            step_idx=step_idx,
        )

        with ref_attention_context(
            mode="r",
            ref_dict=ref_dict,
            is_cfg_guidance=is_cfg_guidance,
            cfg_layout=cfg_layout,
            runtime_checks=runtime_checks,
        ):
            out = self.unet(sample, timestep, encoder_hidden_states, *args, **kwargs)

        if len(ref_dict) > 0:
            _runtime_log(
                f"read pass finished with {len(ref_dict)} leftover ref_dict entries",
                step_idx=step_idx,
                level="WARN",
            )
        else:
            _runtime_log("read pass consumed the timestep-local ref_dict", step_idx=step_idx)
        return out


def ensure_ref_unet_wrapped(pipe) -> None:
    if not hasattr(pipe, "unet"):
        raise ValueError("Pipeline has no `unet` attribute; cannot enable reference attention.")
    if not hasattr(pipe, "scheduler"):
        raise ValueError("Pipeline has no `scheduler` attribute; cannot enable reference attention.")

    if isinstance(pipe.unet, RefOnlyNoisedUNet):
        _require_4_channel_unet(pipe.unet.unet)
        enable_reference_attention(pipe.unet.unet)
        return

    _require_4_channel_unet(pipe.unet)
    enable_reference_attention(pipe.unet)
    pipe.unet = RefOnlyNoisedUNet(pipe.unet, pipe.scheduler)


def debug_refattn(pipe) -> None:
    try:
        unet_obj = getattr(pipe, "unet", None)
        base_unet = getattr(unet_obj, "unet", unet_obj)
        print("[DBG][RefAttn] pipe.unet =", type(unet_obj).__name__)
        print("[DBG][RefAttn] base_unet =", type(base_unet).__name__)

        procs = getattr(base_unet, "attn_processors", None)
        if procs is None:
            print("[DBG][RefAttn] base_unet has no attn_processors")
            return

        wrapped = 0
        enabled = 0
        for proc in procs.values():
            if isinstance(proc, ReferenceOnlyAttnProc):
                wrapped += 1
                enabled += int(bool(proc.enabled))

        print("[DBG][RefAttn] wrapped processors =", wrapped)
        print("[DBG][RefAttn] enabled self-attn processors =", enabled)
    except Exception as exc:
        print("[DBG][RefAttn] debug_refattn failed:", repr(exc))
