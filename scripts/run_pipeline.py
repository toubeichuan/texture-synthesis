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
        "--stage",
        choices=["texture", "prepare_gs", "train_gs", "render_gs", "all"],
        default="all",
        help="Pipeline stage to run.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument(
        "--train-iterations",
        type=int,
        default=None,
        help="Override gaussian.iterations for a quick smoke test or full run.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


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
        "--camera-angle-x",
        str(prepare["camera_angle_x"]),
        "--rotate-x-deg",
        str(prepare.get("rotate_x_deg", 90.0)),
        "--rotate-y-deg",
        str(prepare.get("rotate_y_deg", 0.0)),
        "--rotate-z-deg",
        str(prepare.get("rotate_z_deg", 0.0)),
        "--test-mode",
        str(prepare.get("test_mode", "copy-train")),
    ]
    if prepare.get("max_images", 0):
        command += ["--max-images", str(prepare["max_images"])]
    if prepare.get("copy_metadata", True):
        command.append("--copy-metadata")
    if prepare.get("clean", False):
        command.append("--clean")
    if dry_run:
        command.append("--dry-run")
    run_command(command, cwd=PROJECT_ROOT, dry_run=False)


def run_train_gs(config: Dict[str, Any], dry_run: bool) -> None:
    gaussian = config["gaussian"]
    repo = resolve(gaussian.get("repo", "gaussian-mesh-splatting"))
    env = gaussian.get("env", "gaussian_splatting_mesh")
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
    run_command(command, cwd=repo, dry_run=dry_run)


def main() -> None:
    args = parse_args()
    config = load_config(resolve(args.config))
    if args.train_iterations is not None:
        config.setdefault("gaussian", {})["iterations"] = args.train_iterations
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
