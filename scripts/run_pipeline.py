#!/usr/bin/env python3
"""Configuration-driven texture-transfer pipeline. / 配置驱动的通用纹理迁移流水线。

English:
    Run texture generation, camera conversion, dataset preparation, Gaussian
    training, and rendering in one command. Object names, reference images,
    camera presets, and output settings come from a JSON config; this file does
    not contain experiment-specific names such as Labubu or cat.

中文：
    用一条命令串联纹理生成、相机转换、数据集准备、Gaussian 训练和渲染。
    OBJ、参考图、相机预设和输出参数全部由 JSON 配置提供，本文件不再
    绑定 Labubu、cat 等具体实验名称。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """解析通用配置、自定义运行名和断点续训选项。 / Parse config, run name, and resume options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="流水线 JSON 配置。 / Pipeline JSON config.")
    parser.add_argument("--run-name", default="", help="覆盖自动运行名。 / Override the automatic run name.")
    parser.add_argument("--resume", action="store_true", help="继续已完成数据准备的任务。 / Resume after dataset preparation.")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def resolve(path: str | Path, base: Path = PROJECT_ROOT) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else base / value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_conda_executable() -> str:
    """寻找可用的 conda，避免 nohup 环境缺少 PATH。 / Locate conda even when nohup has a minimal PATH."""
    candidates = [
        os.environ.get("CONDA_EXE", ""),
        shutil.which("conda") or "",
        str(Path(sys.executable).with_name("conda")),
        str(Path.home() / "miniconda3" / "bin" / "conda"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise FileNotFoundError("could not locate a conda executable")


def conda_command(env_name: str, command: Sequence[str]) -> List[str]:
    """构建无缓冲的单环境命令。 / Build an unbuffered command for the shared conda environment."""
    return [
        find_conda_executable(),
        "run",
        "--no-capture-output",
        "-n",
        env_name,
        "env",
        "PYTHONNOUSERSITE=1",
        "PYTHONUNBUFFERED=1",
        *command,
    ]


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    quiet: bool = False,
) -> None:
    """
    实时转发子进程输出到主日志和阶段日志。
    Stream subprocess output to both the main console log and the stage log.
    """
    started_at = datetime.now()
    started_monotonic = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] cwd={cwd}", flush=True)
    print(f"[run] log={log_path}", flush=True)
    print(f"[run] started_at={started_at.isoformat(timespec='seconds')}", flush=True)
    if not quiet:
        print("[run] " + " ".join(str(part) for part in command), flush=True)
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(f"[run] pid={process.pid}", flush=True)
        if process.stdout is None:
            raise RuntimeError("failed to capture subprocess output")
        # 使用二进制块保留 tqdm 的回车刷新等控制字符。
        # Binary chunks preserve carriage-return updates emitted by tqdm.
        while True:
            chunk = process.stdout.read1(4096)
            if not chunk:
                break
            log.write(chunk)
            log.flush()
            stdout_buffer = getattr(sys.stdout, "buffer", None)
            if stdout_buffer is not None:
                stdout_buffer.write(chunk)
                stdout_buffer.flush()
            else:
                print(chunk.decode("utf-8", errors="replace"), end="", flush=True)
        return_code = process.wait()
        elapsed = time.monotonic() - started_monotonic
        print(f"[run] return_code={return_code} elapsed_seconds={elapsed:.1f}", flush=True)
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, list(command))


def numeric_images(directory: Path) -> List[Path]:
    return sorted(
        (path for path in directory.glob("*.png") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )


def find_texture_run(texture_root: Path) -> Path:
    """定位本次新建的唯一纹理输出目录。 / Locate the single fresh texture result directory."""
    candidates = sorted(path.parent for path in texture_root.rglob("args.json"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one fresh texture run under {texture_root}, found {len(candidates)}"
        )
    return candidates[0]


def require_fresh(paths: Iterable[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("fresh run paths already exist: " + ", ".join(existing))


def build_default_run_name(config: Dict[str, Any], now: datetime) -> str:
    """生成 `参考图_OBJ_YYYYMMDD_HHMM` 运行名。 / Build `reference_OBJ_YYYYMMDD_HHMM`."""
    texture = config["texture"]
    reference_name = Path(texture["reference_image"]).stem
    obj_name = Path(texture["obj_file"]).stem
    if not reference_name or not obj_name:
        raise ValueError("reference image and OBJ filenames must have non-empty stems")
    return f"{reference_name}_{obj_name}_{now:%Y%m%d_%H%M}"


def set_stage(manifest_path: Path, manifest: Dict[str, Any], stage: str, status: str) -> None:
    """持久化阶段状态，用于查看进度和断点续训。 / Persist stage status for progress and resume."""
    updated_at = datetime.now().isoformat(timespec="seconds")
    manifest["stages"][stage] = {
        "status": status,
        "updated_at": updated_at,
    }
    dump_json(manifest_path, manifest)
    print(f"[stage] name={stage} status={status} updated_at={updated_at}", flush=True)


def texture_command(config: Dict[str, Any], output_root: Path) -> List[str]:
    """从配置构建 DYC 纹理生成命令。 / Build the DYC texture-generation command from config."""
    texture = config["texture"]
    return [
        "python",
        texture["script"],
        "--input_dir",
        texture["input_dir"],
        "--output_dir",
        str(output_root),
        "--obj_file",
        texture["obj_file"],
        "--prompt",
        texture["prompt"],
        "--style_img",
        texture["reference_image"],
        "--style_mask",
        texture["reference_mask"],
        "--style_img_bg_color",
        "255",
        "255",
        "255",
        "--ip_adapter_path",
        "./ip_adapter",
        "--ip_adapter_strength",
        str(texture["ip_adapter_strength"]),
        "--ip_adapter_n_tokens",
        "16",
        "--controlnet_cond",
        texture["controlnet_cond"],
        "--controlnet_strength",
        str(texture["controlnet_strength"]),
        "--use_cc_edges",
        "True",
        "--use_depth_edges",
        "True",
        "--use_normal_edges",
        "True",
        "--add_view_to_prompt",
        "--ddim_steps",
        str(texture["ddim_steps"]),
        "--guidance_scale",
        str(texture["guidance_scale"]),
        "--new_strength",
        str(texture["new_strength"]),
        "--update_strength",
        str(texture["update_strength"]),
        "--view_threshold",
        str(texture["view_threshold"]),
        "--blend",
        "0",
        "--dist",
        str(texture["dist"]),
        "--num_viewpoints",
        str(texture["num_viewpoints"]),
        "--viewpoint_mode",
        "predefined",
        "--use_principle",
        "--update_steps",
        str(texture["update_steps"]),
        "--update_mode",
        "heuristic",
        "--seed",
        str(texture["seed"]),
        "--tex_resolution",
        texture["texture_resolution"],
        "--camera_preset",
        config["camera"]["preset"],
    ]


def convert_cameras(
    config: Dict[str, Any],
    env_name: str,
    camera_dir: Path,
    logs_dir: Path,
) -> Path:
    """导出相机位姿并转成 GaMeS 数据格式。 / Export poses and convert them to the GaMeS dataset format."""
    camera = config["camera"]
    camera_dir.mkdir(parents=True, exist_ok=False)
    poses_path = camera_dir / "camera_poses.json"
    transforms_path = camera_dir / "transforms_train.json"

    export_command = conda_command(
        env_name,
        [
            "python",
            "scripts/camera/export_camera_poses.py",
            "--viewpoint_mode",
            "predefined",
            "--preset",
            camera["preset"],
            "--dist",
            str(camera["dist"]),
            "--out",
            str(poses_path),
        ],
    )
    run_command(export_command, cwd=PROJECT_ROOT, log_path=logs_dir / "camera_export.log")

    convert_command = conda_command(
        env_name,
        [
            "python",
            "scripts/camera/camera_poses_to_transforms.py",
            "--input",
            str(poses_path),
            "--out",
            str(transforms_path),
            "--camera-angle-x",
            str(camera["camera_angle_x"]),
            "--file-prefix",
            "./train",
            "--file-mode",
            "index",
            "--rotate-x-deg",
            str(camera["rotate_x_deg"]),
            "--rotate-y-deg",
            str(camera["rotate_y_deg"]),
            "--rotate-z-deg",
            str(camera["rotate_z_deg"]),
        ],
    )
    run_command(convert_command, cwd=PROJECT_ROOT, log_path=logs_dir / "camera_convert.log")
    return transforms_path


def camera_signature(frame: Dict[str, Any], precision: int = 8) -> tuple[float, ...]:
    return tuple(
        round(float(value), precision)
        for row in frame["transform_matrix"]
        for value in row
    )


def prepare_dataset(
    config: Dict[str, Any],
    texture_run_dir: Path,
    transforms_path: Path,
    scene_dir: Path,
) -> Dict[str, Any]:
    """
    用 mask/相似度过滤图像，处理极点旋转，并生成训练数据集。
    Filter images with masks/similarity, correct pole rotation, and build the training dataset.
    """
    import numpy as np
    import trimesh
    from PIL import Image, ImageFilter

    dataset = config["dataset"]
    image_dir = texture_run_dir / "generate" / "inpainted"
    mask_dir = texture_run_dir / "generate" / "mask"
    similarity_dir = texture_run_dir / "generate" / "similarity"
    transforms = load_json(transforms_path)
    source_frames = transforms.get("frames", [])
    expected = len(source_frames)
    images = numeric_images(image_dir)
    expected_names = [f"{index}.png" for index in range(expected)]
    if [path.name for path in images] != expected_names:
        raise RuntimeError(
            f"camera preset has {expected} views, but generated images are {[path.name for path in images]}"
        )
    if len(source_frames) != expected:
        raise RuntimeError(f"expected {expected} camera frames, got {len(source_frames)}")
    if len({camera_signature(frame) for frame in source_frames}) != expected:
        raise RuntimeError("generation cameras are not unique")
    pose_views = load_json(transforms_path.parent / "camera_poses.json").get("views", [])
    if len(pose_views) != expected:
        raise RuntimeError(f"expected {expected} pose records, got {len(pose_views)}")
    rotate_poles = bool(dataset.get("rotate_pole_views_180", False))
    pole_tolerance = float(dataset.get("pole_elevation_tolerance_deg", 1e-6))

    scene_dir.mkdir(parents=True, exist_ok=False)
    train_dir = scene_dir / "train"
    train_dir.mkdir()
    retained_frames: List[Dict[str, Any]] = []
    view_records: List[Dict[str, Any]] = []
    threshold = int(round(float(dataset["similarity_threshold"]) * 255.0))
    background = np.array(dataset["background_rgb"], dtype=np.uint8)

    for index, image_path in enumerate(images):
        elevation = float(pose_views[index]["elev"])
        rotation_degrees = (
            180
            if rotate_poles and abs(abs(elevation) - 90.0) <= pole_tolerance
            else 0
        )
        masks = []
        for suffix in ("old", "update", "new"):
            mask_path = mask_dir / f"{index}_{suffix}.png"
            if not mask_path.exists():
                raise FileNotFoundError(mask_path)
            masks.append(np.array(Image.open(mask_path).convert("L")) > 127)
        foreground = np.logical_or.reduce(masks)

        similarity_path = similarity_dir / f"{index}.png"
        if not similarity_path.exists():
            raise FileNotFoundError(similarity_path)
        similarity = np.array(Image.open(similarity_path).convert("L"))
        valid = foreground & (similarity >= threshold)
        foreground_area = int(foreground.sum())
        valid_area = int(valid.sum())
        valid_ratio = valid_area / max(foreground_area, 1)
        accepted = foreground_area > 0 and valid_ratio >= float(dataset["minimum_valid_foreground_ratio"])

        record = {
            "source_index": index,
            "source_image": str(image_path),
            "foreground_pixels": foreground_area,
            "valid_pixels": valid_area,
            "valid_foreground_ratio": valid_ratio,
            "accepted": accepted,
            "elevation": elevation,
            "rotation_degrees": rotation_degrees,
        }
        print(
            "[dataset] source_view={:02d} elev={:+.1f} foreground={} valid={} "
            "valid_ratio={:.4f} accepted={} rotation={}deg".format(
                index,
                elevation,
                foreground_area,
                valid_area,
                valid_ratio,
                accepted,
                rotation_degrees,
            ),
            flush=True,
        )
        if not accepted:
            view_records.append(record)
            continue

        mask_image = Image.fromarray((valid.astype(np.uint8) * 255), mode="L")
        dilate = int(dataset["mask_dilate_pixels"])
        erode = int(dataset["mask_erode_pixels"])
        if dilate > 0:
            mask_image = mask_image.filter(ImageFilter.MaxFilter(dilate * 2 + 1))
        if erode > 0:
            mask_image = mask_image.filter(ImageFilter.MinFilter(erode * 2 + 1))
        final_mask = np.array(mask_image) > 127

        source_rgb = np.array(Image.open(image_path).convert("RGB"))
        if rotation_degrees == 180:
            source_rgb = np.rot90(source_rgb, 2).copy()
            final_mask = np.rot90(final_mask, 2).copy()
        cleaned = np.empty_like(source_rgb)
        cleaned[:] = background
        cleaned[final_mask] = source_rgb[final_mask]

        output_index = len(retained_frames)
        output_image = train_dir / f"{output_index}.png"
        Image.fromarray(cleaned, mode="RGB").save(output_image)
        frame = json.loads(json.dumps(source_frames[index]))
        frame["file_path"] = f"./train/{output_index}"
        retained_frames.append(frame)
        record["output_index"] = output_index
        record["output_image"] = str(output_image)
        view_records.append(record)

    minimum_views = int(dataset["minimum_retained_views"])
    if len(retained_frames) < minimum_views:
        raise RuntimeError(
            f"mask/similarity filtering retained {len(retained_frames)} views; minimum is {minimum_views}"
        )
    if len({camera_signature(frame) for frame in retained_frames}) != len(retained_frames):
        raise RuntimeError("retained cameras are not unique")

    source_mesh = resolve(dataset["source_mesh"])
    source = trimesh.load(source_mesh, force="mesh", process=False)
    bounds = source.bounds.copy()
    center = (bounds[0] + bounds[1]) / 2.0
    max_extent = float((bounds[1] - bounds[0]).max())
    if max_extent <= 0:
        raise RuntimeError("source mesh has invalid bounds")
    normalized_vertices = (source.vertices - center) / max_extent
    normalized_mesh = trimesh.Trimesh(
        vertices=normalized_vertices,
        faces=source.faces,
        process=False,
    )
    normalized_mesh.export(scene_dir / "mesh.obj")

    output_transforms = {
        "camera_angle_x": float(transforms["camera_angle_x"]),
        "frames": retained_frames,
    }
    dump_json(scene_dir / "transforms_train.json", output_transforms)
    dump_json(scene_dir / "transforms_test.json", output_transforms)
    dump_json(scene_dir / "transforms_val.json", output_transforms)
    shutil.copy2(transforms_path.parent / "camera_poses.json", scene_dir / "camera_poses.json")

    dataset_manifest = {
        "source_texture_run": str(texture_run_dir),
        "source_mesh": str(source_mesh),
        "source_mesh_sha256": sha256(source_mesh),
        "mesh_transform": "(vertices - bbox_center) / max_bbox_extent",
        "source_mesh_bounds": bounds.tolist(),
        "source_mesh_center": center.tolist(),
        "source_mesh_max_extent": max_extent,
        "generated_view_count": len(images),
        "retained_view_count": len(retained_frames),
        "rejected_view_count": len(images) - len(retained_frames),
        "rotate_pole_views_180": rotate_poles,
        "views": view_records,
    }
    dump_json(scene_dir / "dataset_manifest.json", dataset_manifest)
    print(
        f"[dataset] completed generated={len(images)} retained={len(retained_frames)} "
        f"rejected={len(images) - len(retained_frames)} scene_dir={scene_dir}",
        flush=True,
    )
    return dataset_manifest


def train_gaussians(
    config: Dict[str, Any],
    env_name: str,
    scene_dir: Path,
    model_dir: Path,
    logs_dir: Path,
) -> None:
    """训练基于 mesh 的 Gaussian Splatting。 / Train mesh-based Gaussian Splatting."""
    gaussian = config["gaussian"]
    repo = resolve(gaussian["repo"])
    print(f"[training] tensorboard_logdir={model_dir}", flush=True)
    command = [
        "env",
        "MKL_THREADING_LAYER=GNU",
        "python",
        "train.py",
        "--eval",
        "-s",
        str(scene_dir),
        "-m",
        str(model_dir),
        "--gs_type",
        gaussian["gs_type"],
        "--num_splats",
        str(gaussian["num_splats"]),
        "--iterations",
        str(gaussian["iterations"]),
    ]
    if gaussian.get("white_background", True):
        command.append("-w")
    run_command(
        conda_command(env_name, command),
        cwd=repo,
        log_path=logs_dir / "gaussian_train.log",
        quiet=True,
    )


def render_gaussians(
    config: Dict[str, Any],
    env_name: str,
    model_dir: Path,
    render_dir: Path,
    logs_dir: Path,
) -> None:
    """渲染训练集和测试集，并收集结果。 / Render train/test cameras and collect outputs."""
    gaussian = config["gaussian"]
    render = config["render"]
    repo = resolve(gaussian["repo"])
    command = [
        "env",
        "MKL_THREADING_LAYER=GNU",
        f"PYTHONPATH={repo}",
        "python",
        "scripts/render.py",
        "-m",
        str(model_dir),
        "--gs_type",
        gaussian["gs_type"],
        "--iteration",
        str(gaussian["iterations"]),
    ]
    if not render.get("render_train", True):
        command.append("--skip_train")
    if not render.get("render_test", True):
        command.append("--skip_test")
    run_command(
        conda_command(env_name, command),
        cwd=repo,
        log_path=logs_dir / "gaussian_render.log",
    )
    render_dir.mkdir(parents=True, exist_ok=False)
    for name in ("train", "test"):
        source = model_dir / name
        if source.exists():
            shutil.copytree(source, render_dir / name)


def main() -> None:
    """按阶段执行完整流水线并写入 manifest。 / Execute all stages and maintain the run manifest."""
    args = parse_args()
    config_path = resolve(args.config)
    config = load_json(config_path)
    now = datetime.now()
    run_name = args.run_name or build_default_run_name(config, now)

    env_name = config["env"]
    run_root = PROJECT_ROOT / "outputs" / "pipeline_runs" / run_name
    texture_root = PROJECT_ROOT / "outputs" / "texture_synthesis" / run_name
    scene_dir = PROJECT_ROOT / "data" / "scenes" / run_name
    model_dir = PROJECT_ROOT / "outputs" / "gaussian_splatting" / run_name
    render_dir = PROJECT_ROOT / "outputs" / "renders" / run_name
    logs_dir = run_root / "logs"
    camera_dir = run_root / "camera"
    manifest_path = run_root / "run_manifest.json"

    texture_cwd = resolve(config["texture"]["cwd"])
    if args.resume:
        if not args.run_name:
            raise ValueError("--resume requires --run-name")
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = load_json(manifest_path)
        for stage in ("texture", "camera", "dataset"):
            if manifest.get("stages", {}).get(stage, {}).get("status") != "completed":
                raise RuntimeError(f"cannot resume: {stage} stage is not completed")
        texture_run_dir = Path(manifest["paths"]["texture_run_dir"])
        transforms_path = camera_dir / "transforms_train.json"
        for required in (texture_run_dir, transforms_path, scene_dir):
            if not required.exists():
                raise FileNotFoundError(required)
        manifest.pop("failed_stage", None)
        manifest.pop("error", None)
        manifest.pop("traceback", None)
        manifest["status"] = "resumed"
        manifest["resumed_at"] = now.isoformat(timespec="seconds")
        dump_json(manifest_path, manifest)
    else:
        require_fresh((run_root, texture_root, scene_dir, model_dir, render_dir))
        run_root.mkdir(parents=True, exist_ok=False)
        input_mesh = texture_cwd / config["texture"]["input_dir"] / config["texture"]["obj_file"]
        reference_image = texture_cwd / config["texture"]["reference_image"]
        reference_mask = texture_cwd / config["texture"]["reference_mask"]
        for required in (input_mesh, reference_image, reference_mask):
            if not required.is_file():
                raise FileNotFoundError(required)

        manifest = {
            "run_name": run_name,
            "created_at": now.isoformat(timespec="seconds"),
            "config": str(config_path),
            "environment": env_name,
            "paths": {
                "run_root": str(run_root),
                "texture_root": str(texture_root),
                "scene_dir": str(scene_dir),
                "model_dir": str(model_dir),
                "render_dir": str(render_dir),
            },
            "inputs": {
                "mesh": {"path": str(input_mesh), "sha256": sha256(input_mesh)},
                "reference_image": {"path": str(reference_image), "sha256": sha256(reference_image)},
                "reference_mask": {"path": str(reference_mask), "sha256": sha256(reference_mask)},
            },
            "stages": {},
        }
        dump_json(run_root / "resolved_config.json", config)
        dump_json(manifest_path, manifest)
    print(f"RUN_NAME={run_name}", flush=True)
    print(f"[pipeline] config={config_path}", flush=True)
    print(f"[pipeline] environment={env_name}", flush=True)
    print(f"[pipeline] reference_image={config['texture']['reference_image']}", flush=True)
    print(f"[pipeline] reference_mask={config['texture']['reference_mask']}", flush=True)
    print(f"[pipeline] obj_file={config['texture']['obj_file']}", flush=True)
    print(f"[pipeline] camera_preset={config['camera']['preset']}", flush=True)
    print(f"[pipeline] run_root={run_root}", flush=True)
    print(f"[pipeline] texture_root={texture_root}", flush=True)
    print(f"[pipeline] scene_dir={scene_dir}", flush=True)
    print(f"[pipeline] model_dir={model_dir}", flush=True)
    print(f"[pipeline] render_dir={render_dir}", flush=True)

    current_stage = "training" if args.resume else "texture"
    try:
        if not args.resume:
            set_stage(manifest_path, manifest, current_stage, "running")
            run_command(
                conda_command(env_name, texture_command(config, texture_root)),
                cwd=texture_cwd,
                log_path=logs_dir / "texture_generation.log",
            )
            texture_run_dir = find_texture_run(texture_root)
            images = numeric_images(texture_run_dir / "generate" / "inpainted")
            if not images:
                raise RuntimeError("texture stage did not generate any numeric view images")
            manifest["paths"]["texture_run_dir"] = str(texture_run_dir)
            set_stage(manifest_path, manifest, current_stage, "completed")

            current_stage = "camera"
            set_stage(manifest_path, manifest, current_stage, "running")
            transforms_path = convert_cameras(config, env_name, camera_dir, logs_dir)
            set_stage(manifest_path, manifest, current_stage, "completed")

            current_stage = "dataset"
            set_stage(manifest_path, manifest, current_stage, "running")
            dataset_manifest = prepare_dataset(config, texture_run_dir, transforms_path, scene_dir)
            manifest["dataset"] = {
                "generated_views": dataset_manifest["generated_view_count"],
                "retained_views": dataset_manifest["retained_view_count"],
                "rejected_views": dataset_manifest["rejected_view_count"],
            }
            set_stage(manifest_path, manifest, current_stage, "completed")

        current_stage = "training"
        if manifest.get("stages", {}).get(current_stage, {}).get("status") != "completed":
            if model_dir.exists():
                raise FileExistsError(f"refusing to overwrite partial model directory: {model_dir}")
            set_stage(manifest_path, manifest, current_stage, "running")
            train_gaussians(config, env_name, scene_dir, model_dir, logs_dir)
        checkpoint = model_dir / "point_cloud" / f"iteration_{config['gaussian']['iterations']}" / "point_cloud.ply"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        set_stage(manifest_path, manifest, current_stage, "completed")

        current_stage = "render"
        if manifest.get("stages", {}).get(current_stage, {}).get("status") != "completed":
            if render_dir.exists():
                raise FileExistsError(f"refusing to overwrite partial render directory: {render_dir}")
            set_stage(manifest_path, manifest, current_stage, "running")
            render_gaussians(config, env_name, model_dir, render_dir, logs_dir)
        set_stage(manifest_path, manifest, current_stage, "completed")
        manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
        manifest["status"] = "completed"
        dump_json(manifest_path, manifest)
        print(f"PIPELINE_COMPLETED={run_name}", flush=True)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failed_stage"] = current_stage
        manifest["error"] = str(error)
        manifest["traceback"] = traceback.format_exc()
        set_stage(manifest_path, manifest, current_stage, "failed")
        dump_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
