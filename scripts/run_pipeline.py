#!/usr/bin/env python3
"""Project-level pipeline runner.

This runner orchestrates the research pipeline without importing the heavy
texture-synthesis or Gaussian Splatting packages. Each stage is executed as a
subprocess so different conda environments can be used safely.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run texture synthesis -> GS scene preparation -> GaMeS stages.")
    parser.add_argument("--config", required=True, help="Pipeline config JSON file.")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Convenience name for this experiment. Sets scene/output paths unless they are overridden explicitly.",
    )
    parser.add_argument(
        "--stage",
        choices=["texture", "prepare_gs", "train_gs", "render_gs", "all"],
        default="all",
        help="Pipeline stage to run.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")

    prepare_group = parser.add_argument_group("GS scene preparation overrides")
    prepare_group.add_argument("--scene-root", default="data/scenes", help="Root used with --run-name for prepared GS scenes.")
    prepare_group.add_argument("--texture-output", default=None, help="Override prepare_gs.texture_output.")
    prepare_group.add_argument("--scene-dir", default=None, help="Override prepare_gs.scene_dir and gaussian.source_path.")
    prepare_group.add_argument("--mesh-source", default=None, help="Override prepare_gs.mesh_source.")
    prepare_group.add_argument("--image-dir", default=None, help="Override prepare_gs.image_dir.")
    prepare_group.add_argument("--image-pattern", default=None, help="Override prepare_gs.image_pattern.")
    prepare_group.add_argument("--image-filter", choices=["numeric", "all"], default=None, help="Override prepare_gs.image_filter.")
    prepare_group.add_argument("--max-images", type=int, default=None, help="Override prepare_gs.max_images.")
    prepare_group.add_argument("--viewpoints", default=None, help="Override prepare_gs.viewpoints.")
    prepare_group.add_argument("--train-transforms", default=None, help="Use an existing transforms_train.json for training cameras.")
    prepare_group.add_argument("--test-transforms", default=None, help="Use an existing transforms JSON for test cameras.")
    prepare_group.add_argument("--val-transforms", default=None, help="Use an existing transforms JSON for validation cameras.")
    prepare_group.add_argument("--camera-angle-x", type=float, default=None, help="Override prepare_gs.camera_angle_x.")
    prepare_group.add_argument("--rotate-x-deg", type=float, default=None, help="Override prepare_gs.rotate_x_deg.")
    prepare_group.add_argument("--rotate-y-deg", type=float, default=None, help="Override prepare_gs.rotate_y_deg.")
    prepare_group.add_argument("--rotate-z-deg", type=float, default=None, help="Override prepare_gs.rotate_z_deg.")
    prepare_group.add_argument("--test-mode", choices=["copy-train", "empty"], default=None, help="Override prepare_gs.test_mode.")
    prepare_group.add_argument("--allow-repeat-cameras", action="store_true", help="Allow repeated training camera poses.")
    prepare_group.add_argument("--no-clean", action="store_true", help="Do not clean the scene directory before preparation.")

    gaussian_group = parser.add_argument_group("Gaussian training/rendering overrides")
    gaussian_group.add_argument(
        "--gaussian-output-root",
        default="outputs/gaussian_splatting",
        help="Root used with --run-name for GaMeS checkpoints and renders.",
    )
    gaussian_group.add_argument("--source-path", default=None, help="Override gaussian.source_path.")
    gaussian_group.add_argument("--model-path", default=None, help="Override gaussian.model_path.")
    gaussian_group.add_argument("--gs-type", default=None, help="Override gaussian.gs_type.")
    gaussian_group.add_argument("--num-splats", type=int, default=None, help="Override gaussian.num_splats.")
    parser.add_argument(
        "--train-iterations",
        type=int,
        default=None,
        help="Override gaussian.iterations for a quick smoke test or full run.",
    )
    gaussian_group.add_argument("--white-background", action="store_true", help="Force white background.")
    gaussian_group.add_argument("--black-background", action="store_true", help="Force black background.")
    gaussian_group.add_argument("--render-iteration", type=int, default=None, help="Render a specific saved iteration.")
    gaussian_group.add_argument("--skip-train-render", action="store_true", help="Skip rendering train cameras.")
    gaussian_group.add_argument("--skip-test-render", action="store_true", help="Skip rendering test cameras.")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
    prepare = config.setdefault("prepare_gs", {})
    gaussian = config.setdefault("gaussian", {})

    # The JSON file holds reproducible defaults. CLI overrides are for the
    # last-mile experiment knobs that change frequently while iterating.
    prepare_overrides = {
        "texture_output": args.texture_output,
        "scene_dir": args.scene_dir,
        "mesh_source": args.mesh_source,
        "image_dir": args.image_dir,
        "image_pattern": args.image_pattern,
        "image_filter": args.image_filter,
        "max_images": args.max_images,
        "viewpoints": args.viewpoints,
        "train_transforms": args.train_transforms,
        "test_transforms": args.test_transforms,
        "val_transforms": args.val_transforms,
        "camera_angle_x": args.camera_angle_x,
        "rotate_x_deg": args.rotate_x_deg,
        "rotate_y_deg": args.rotate_y_deg,
        "rotate_z_deg": args.rotate_z_deg,
        "test_mode": args.test_mode,
    }
    for key, value in prepare_overrides.items():
        if value is not None:
            prepare[key] = value
    if args.allow_repeat_cameras:
        prepare["allow_repeat_cameras"] = True
    if args.no_clean:
        prepare["clean"] = False

    gaussian_overrides = {
        "source_path": args.source_path,
        "model_path": args.model_path,
        "gs_type": args.gs_type,
        "num_splats": args.num_splats,
        "iterations": args.train_iterations,
        "render_iteration": args.render_iteration,
    }
    for key, value in gaussian_overrides.items():
        if value is not None:
            gaussian[key] = value
    if args.white_background:
        gaussian["white_background"] = True
    if args.black_background:
        gaussian["white_background"] = False
    if args.skip_train_render:
        gaussian["skip_train_render"] = True
    if args.skip_test_render:
        gaussian["skip_test_render"] = True

    # A run name gives every experiment its own prepared scene and output
    # folder, so smoke tests and full runs do not overwrite each other.
    if args.run_name:
        scene_dir = str(Path(args.scene_root) / args.run_name)
        model_path = str(Path(args.gaussian_output_root) / args.run_name)
        if args.scene_dir is None:
            prepare["scene_dir"] = scene_dir
        if args.source_path is None:
            gaussian["source_path"] = prepare["scene_dir"]
        if args.model_path is None:
            gaussian["model_path"] = model_path

    if args.scene_dir and not args.source_path:
        gaussian["source_path"] = args.scene_dir


def run_command(command: Sequence[str], cwd: Path | None = None, dry_run: bool = False) -> None:
    display = " ".join(str(part) for part in command)
    if cwd:
        print(f"[cwd] {cwd}", flush=True)
    print(f"[cmd] {display}", flush=True)
    if dry_run:
        return
    subprocess.run(list(command), cwd=str(cwd) if cwd else None, check=True)


def conda_prefix(env_name: str | None) -> List[str]:
    return ["conda", "run", "-n", env_name] if env_name else []


def run_texture(config: Dict[str, Any], dry_run: bool) -> None:
    texture = config.get("texture", {})
    command = texture.get("command")
    if not command:
        print("[texture] no texture.command configured; skipping texture synthesis")
        return
    env = texture.get("env")
    cwd = resolve(texture.get("cwd", "."))
    run_command(conda_prefix(env) + command, cwd=cwd, dry_run=dry_run)


def run_prepare_gs(config: Dict[str, Any], dry_run: bool) -> None:
    prepare = config["prepare_gs"]
    # This stage only prepares a lightweight NeRF/GaMeS-style dataset:
    # mesh.obj, train/*.png, and transforms_*.json. It does not train GS.
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/dataset/prepare_gs_scene.py"),
        "--texture-output",
        str(resolve(prepare["texture_output"])),
        "--scene-dir",
        str(resolve(prepare["scene_dir"])),
        "--mesh-source",
        str(prepare.get("mesh_source", "auto")),
        "--image-dir",
        str(prepare.get("image_dir", "update/inpainted")),
        "--image-pattern",
        str(prepare.get("image_pattern", "*.png")),
        "--image-filter",
        str(prepare.get("image_filter", "numeric")),
        "--viewpoints",
        str(prepare.get("viewpoints", "viewpoints.json")),
        "--rotate-x-deg",
        str(prepare.get("rotate_x_deg", 90.0)),
        "--rotate-y-deg",
        str(prepare.get("rotate_y_deg", 0.0)),
        "--rotate-z-deg",
        str(prepare.get("rotate_z_deg", 0.0)),
        "--test-mode",
        str(prepare.get("test_mode", "copy-train")),
    ]
    if prepare.get("train_transforms"):
        command += ["--train-transforms", str(resolve(prepare["train_transforms"]))]
    if prepare.get("test_transforms"):
        command += ["--test-transforms", str(resolve(prepare["test_transforms"]))]
    if prepare.get("val_transforms"):
        command += ["--val-transforms", str(resolve(prepare["val_transforms"]))]
    if prepare.get("camera_angle_x") is not None:
        command += ["--camera-angle-x", str(prepare["camera_angle_x"])]
    if prepare.get("max_images", 0):
        command += ["--max-images", str(prepare["max_images"])]
    if prepare.get("copy_metadata", True):
        command.append("--copy-metadata")
    if prepare.get("clean", False):
        command.append("--clean")
    if prepare.get("allow_repeat_cameras", False):
        command.append("--allow-repeat-cameras")
    if dry_run:
        command.append("--dry-run")
    run_command(command, cwd=PROJECT_ROOT, dry_run=False)


def run_train_gs(config: Dict[str, Any], dry_run: bool) -> None:
    gaussian = config["gaussian"]
    repo = resolve(gaussian.get("repo", "gaussian-mesh-splatting"))
    env = gaussian.get("env", "gaussian_splatting_mesh")
    # GaMeS expects to be launched from its own repository because it imports
    # local modules such as scene, games, and renderer.
    command = conda_prefix(env) + [
        "python",
        "train.py",
        "--eval",
        "-s",
        str(resolve(gaussian["source_path"])),
        "-m",
        str(resolve(gaussian["model_path"])),
        "--gs_type",
        str(gaussian.get("gs_type", "gs_mesh")),
        "--num_splats",
        str(gaussian.get("num_splats", 5)),
    ]
    if gaussian.get("white_background", True):
        command.append("-w")
    if gaussian.get("iterations"):
        command += ["--iterations", str(gaussian["iterations"])]
    run_command(command, cwd=repo, dry_run=dry_run)


def run_render_gs(config: Dict[str, Any], dry_run: bool) -> None:
    gaussian = config["gaussian"]
    repo = resolve(gaussian.get("repo", "gaussian-mesh-splatting"))
    env = gaussian.get("env", "gaussian_splatting_mesh")
    python_path = str(repo)
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        python_path = f"{python_path}{os.pathsep}{existing_pythonpath}"
    # scripts/render.py lives one directory below the repository root, so we
    # explicitly expose the root on PYTHONPATH before invoking it.
    command = conda_prefix(env) + [
        "env",
        f"PYTHONPATH={python_path}",
        "python",
        "scripts/render.py",
        "-m",
        str(resolve(gaussian["model_path"])),
        "--gs_type",
        str(gaussian.get("gs_type", "gs_mesh")),
    ]
    if gaussian.get("render_iteration") is not None:
        command += ["--iteration", str(gaussian["render_iteration"])]
    if gaussian.get("skip_train_render", False):
        command.append("--skip_train")
    if gaussian.get("skip_test_render", False):
        command.append("--skip_test")
    run_command(command, cwd=repo, dry_run=dry_run)


def main() -> None:
    args = parse_args()
    config = load_config(resolve(args.config))
    apply_cli_overrides(config, args)
    stages = ["texture", "prepare_gs", "train_gs", "render_gs"] if args.stage == "all" else [args.stage]
    for stage in stages:
        print(f"\n=== {stage} ===", flush=True)
        if stage == "texture":
            run_texture(config, args.dry_run)
        elif stage == "prepare_gs":
            run_prepare_gs(config, args.dry_run)
        elif stage == "train_gs":
            run_train_gs(config, args.dry_run)
        elif stage == "render_gs":
            run_render_gs(config, args.dry_run)


if __name__ == "__main__":
    main()
