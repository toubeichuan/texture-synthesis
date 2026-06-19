#!/usr/bin/env python3
"""Prepare a GaMeS/3DGS scene from EASI-DYC texture synthesis outputs.

The script converts a texture-synthesis run directory into the NeRF-style scene
layout expected by gaussian-mesh-splatting:

    data/scenes/<scene>/
    |-- mesh.obj
    |-- train/0.png
    |-- train/1.png
    |-- transforms_train.json
    |-- transforms_test.json

It intentionally avoids importing EASI or Gaussian Splatting code, so it can run
from a lightweight Python environment before the heavy training stage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a mesh-guided GS scene from an EASI-DYC output folder.")
    parser.add_argument("--texture-output", required=True, help="Texture synthesis run directory.")
    parser.add_argument("--scene-dir", required=True, help="Output GS scene directory.")
    parser.add_argument("--mesh-source", default="auto", help='Mesh source path relative to texture output, or "auto".')
    parser.add_argument("--image-dir", default="update/inpainted", help="Image directory relative to texture output.")
    parser.add_argument("--image-pattern", default="*.png", help="Glob used inside --image-dir.")
    parser.add_argument(
        "--image-filter",
        choices=["numeric", "all"],
        default="numeric",
        help="numeric keeps files whose stem is an integer, e.g. 0.png, 1.png.",
    )
    parser.add_argument("--max-images", type=int, default=0, help="Optional maximum number of images to copy.")
    parser.add_argument("--viewpoints", default="viewpoints.json", help="Viewpoints/camera JSON relative to texture output.")
    parser.add_argument("--camera-angle-x", type=float, required=True, help="Horizontal field of view in radians.")
    parser.add_argument("--rotate-x-deg", type=float, default=90.0, help="World X rotation applied to c2w before writing transforms.")
    parser.add_argument("--rotate-y-deg", type=float, default=0.0, help="World Y rotation applied to c2w before writing transforms.")
    parser.add_argument("--rotate-z-deg", type=float, default=0.0, help="World Z rotation applied to c2w before writing transforms.")
    parser.add_argument("--target", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="Look-at target for EASI viewpoints.")
    parser.add_argument(
        "--test-mode",
        choices=["copy-train", "empty"],
        default="copy-train",
        help="How to create transforms_test.json.",
    )
    parser.add_argument("--copy-metadata", action="store_true", help="Copy args.json/viewpoints.json into the scene folder.")
    parser.add_argument("--clean", action="store_true", help="Remove an existing scene directory before writing.")
    parser.add_argument(
        "--allow-repeat-cameras",
        action="store_true",
        help="Allow multiple supervision images to share the same camera pose. Intended only for smoke tests.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the planned conversion without writing files.")
    return parser.parse_args()


def resolve_path(path: str | Path, base: Path = PROJECT_ROOT) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p)


def natural_key(path: Path) -> Tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def numeric_stem(path: Path) -> Optional[int]:
    return int(path.stem) if path.stem.isdigit() else None


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def list_supervision_images(image_dir: Path, pattern: str, image_filter: str, max_images: int) -> List[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory not found: {image_dir}")

    images = [p for p in image_dir.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if image_filter == "numeric":
        images = [p for p in images if numeric_stem(p) is not None]
        images.sort(key=lambda p: int(p.stem))
    else:
        images.sort(key=natural_key)

    if max_images > 0:
        images = images[:max_images]
    if not images:
        raise FileNotFoundError(f"no supervision images matched {pattern!r} in {image_dir}")
    return images


def mesh_score(texture_output: Path, path: Path) -> Tuple[int, int, int, str]:
    rel = path.relative_to(texture_output)
    parts = rel.parts
    stage_score = 0
    if parts[:2] == ("update", "mesh"):
        stage_score = 30
    elif parts[:2] == ("generate", "mesh"):
        stage_score = 20
    elif len(parts) == 1:
        stage_score = 10

    stem = path.stem
    post_score = 1 if stem.endswith("_post") else 0
    match = re.search(r"(\d+)", stem)
    numeric_score = int(match.group(1)) if match else -1
    return (stage_score, post_score, numeric_score, path.name)


def find_mesh(texture_output: Path, mesh_source: str) -> Path:
    if mesh_source != "auto":
        mesh = resolve_path(mesh_source, texture_output)
        if not mesh.exists():
            raise FileNotFoundError(f"mesh source not found: {mesh}")
        return mesh

    candidates: List[Path] = []
    for rel_dir in ("update/mesh", "generate/mesh", "."):
        mesh_dir = texture_output / rel_dir
        if mesh_dir.is_dir():
            candidates.extend(p for p in mesh_dir.glob("*.obj") if p.is_file())

    if not candidates:
        raise FileNotFoundError(f"could not auto-detect a mesh under {texture_output}")
    return max(candidates, key=lambda p: mesh_score(texture_output, p))


def read_mtllibs(obj_path: Path) -> List[str]:
    mtllibs: List[str] = []
    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("mtllib "):
                mtllibs.append(stripped.split(maxsplit=1)[1])
    return mtllibs


def read_mtl_texture_refs(mtl_path: Path) -> List[str]:
    refs: List[str] = []
    if not mtl_path.exists():
        return refs
    with mtl_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("map_Kd "):
                refs.append(stripped.split(maxsplit=1)[1])
    return refs


def copy_mesh_bundle(mesh_source: Path, scene_dir: Path, dry_run: bool) -> List[str]:
    copied: List[str] = ["mesh.obj"]
    if dry_run:
        return copied

    shutil.copy2(mesh_source, scene_dir / "mesh.obj")
    mesh_parent = mesh_source.parent

    for mtl_name in read_mtllibs(mesh_source):
        source_mtl = mesh_parent / mtl_name
        if source_mtl.exists():
            shutil.copy2(source_mtl, scene_dir / source_mtl.name)
            copied.append(source_mtl.name)
            for tex_name in read_mtl_texture_refs(source_mtl):
                source_tex = mesh_parent / tex_name
                if source_tex.exists():
                    shutil.copy2(source_tex, scene_dir / source_tex.name)
                    copied.append(source_tex.name)

    return copied


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def sub(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def norm(v: Sequence[float]) -> float:
    return math.sqrt(dot(v, v))


def normalize(v: Sequence[float]) -> List[float]:
    length = norm(v)
    if length == 0:
        raise ValueError("cannot normalize a zero vector")
    return [x / length for x in v]


def cross(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def camera_position(dist: float, elev_deg: float, azim_deg: float) -> List[float]:
    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)
    return [
        dist * math.cos(elev) * math.sin(azim),
        dist * math.sin(elev),
        dist * math.cos(elev) * math.cos(azim),
    ]


def look_at_c2w(camera_pos: Sequence[float], target: Sequence[float]) -> List[List[float]]:
    up_world = [0.0, 1.0, 0.0]
    z_axis = normalize(sub(camera_pos, target))
    x_axis = normalize(cross(up_world, z_axis))
    y_axis = cross(z_axis, x_axis)
    return [
        [x_axis[0], y_axis[0], z_axis[0], camera_pos[0]],
        [x_axis[1], y_axis[1], z_axis[1], camera_pos[1]],
        [x_axis[2], y_axis[2], z_axis[2], camera_pos[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rotation_x(angle_deg: float) -> List[List[float]]:
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    return [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]]


def rotation_y(angle_deg: float) -> List[List[float]]:
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    return [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]]


def rotation_z(angle_deg: float) -> List[List[float]]:
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    return [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def matmul4(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def pose_rotation(rx: float, ry: float, rz: float) -> List[List[float]]:
    return matmul4(rotation_z(rz), matmul4(rotation_y(ry), rotation_x(rx)))


def expand_easi_viewpoints(data: Dict[str, Any], num_frames: int, target: Sequence[float]) -> List[Dict[str, Any]]:
    if "views" in data:
        views = data["views"]
        if len(views) < num_frames:
            raise ValueError(f"camera_poses JSON has {len(views)} views, but {num_frames} images were selected")
        return views[:num_frames]

    dist_list = [float(v) for v in data.get("dist", [])]
    elev_list = [float(v) for v in data.get("elev", [])]
    azim_list = [float(v) for v in data.get("azim", [])]
    if not (dist_list and elev_list and azim_list):
        raise ValueError("viewpoints JSON must contain either views or dist/elev/azim lists")

    if len(dist_list) == len(elev_list) == len(azim_list) == num_frames:
        indices = list(range(num_frames))
    elif data.get("view"):
        raw_indices = [int(v) for v in data["view"]]
        indices = raw_indices[:num_frames]
        if len(indices) < num_frames:
            if len(raw_indices) == 1:
                indices = raw_indices * num_frames
            else:
                raise ValueError(f"view list has {len(raw_indices)} entries, but {num_frames} images were selected")
    elif len(dist_list) == len(elev_list) == len(azim_list) == 1:
        indices = [0] * num_frames
    else:
        raise ValueError(
            "cannot align viewpoints to images; provide a view list, one camera, or exactly one camera per image"
        )

    views: List[Dict[str, Any]] = []
    for out_idx, camera_idx in enumerate(indices):
        if camera_idx >= len(dist_list) or camera_idx >= len(elev_list) or camera_idx >= len(azim_list):
            raise IndexError(f"view index {camera_idx} is outside available camera lists")
        pos = camera_position(dist_list[camera_idx], elev_list[camera_idx], azim_list[camera_idx])
        views.append(
            {
                "view_idx": camera_idx,
                "source_order": out_idx,
                "dist": dist_list[camera_idx],
                "elev": elev_list[camera_idx],
                "azim": azim_list[camera_idx],
                "camera_pos": pos,
                "target": list(target),
                "c2w": look_at_c2w(pos, target),
            }
        )
    return views


def make_transforms(
    views: List[Dict[str, Any]],
    camera_angle_x: float,
    file_prefix: str,
    rotation: List[List[float]],
) -> Dict[str, Any]:
    frames = []
    turn_rotation = 2.0 * math.pi / len(views) if views else 0.0
    for index, view in enumerate(views):
        frames.append(
            {
                "file_path": f"{file_prefix.rstrip('/')}/{index}",
                "rotation": turn_rotation,
                "transform_matrix": matmul4(rotation, view["c2w"]),
            }
        )
    return {"camera_angle_x": camera_angle_x, "frames": frames}


def count_unique_camera_poses(transforms: Dict[str, Any], precision: int = 8) -> int:
    signatures = set()
    for frame in transforms["frames"]:
        matrix = frame["transform_matrix"]
        signature = tuple(round(value, precision) for row in matrix for value in row)
        signatures.add(signature)
    return len(signatures)


def assert_transforms_paths(scene_dir: Path, transforms: Dict[str, Any]) -> None:
    for frame in transforms["frames"]:
        rel = frame["file_path"]
        rel = rel[2:] if rel.startswith("./") else rel
        image_path = scene_dir / f"{rel}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"transform image path does not exist: {image_path}")


def write_scene(args: argparse.Namespace) -> Dict[str, Any]:
    texture_output = resolve_path(args.texture_output)
    scene_dir = resolve_path(args.scene_dir)
    image_dir = resolve_path(args.image_dir, texture_output)
    viewpoints_path = resolve_path(args.viewpoints, texture_output)

    images = list_supervision_images(image_dir, args.image_pattern, args.image_filter, args.max_images)
    mesh_source = find_mesh(texture_output, args.mesh_source)
    viewpoints = load_json(viewpoints_path)
    views = expand_easi_viewpoints(viewpoints, len(images), args.target)
    rotation = pose_rotation(args.rotate_x_deg, args.rotate_y_deg, args.rotate_z_deg)

    train_transforms = make_transforms(views, args.camera_angle_x, "./train", rotation)
    test_transforms = train_transforms if args.test_mode == "copy-train" else {"camera_angle_x": args.camera_angle_x, "frames": []}
    unique_camera_poses = count_unique_camera_poses(train_transforms)
    if len(images) > 1 and unique_camera_poses < len(images) and not args.allow_repeat_cameras:
        raise ValueError(
            "selected supervision images do not have one unique camera pose per image "
            f"({len(images)} images, {unique_camera_poses} unique camera poses). "
            "This usually means an EASI-DYC locked-view/update-step output was selected instead of a multi-view output. "
            "Use a multi-view viewpoints.json, reduce --max-images to 1 for a smoke test, or pass "
            "--allow-repeat-cameras explicitly if repeated poses are intentional."
        )

    plan = {
        "texture_output": str(texture_output),
        "scene_dir": str(scene_dir),
        "mesh_source": str(mesh_source),
        "image_dir": str(image_dir),
        "num_images": len(images),
        "first_image": str(images[0]),
        "last_image": str(images[-1]),
        "unique_camera_poses": unique_camera_poses,
        "camera_angle_x": args.camera_angle_x,
        "rotate_xyz_deg": [args.rotate_x_deg, args.rotate_y_deg, args.rotate_z_deg],
        "test_mode": args.test_mode,
    }

    if args.dry_run:
        return plan

    if scene_dir.exists() and args.clean:
        shutil.rmtree(scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)
    train_dir = scene_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    copied_mesh = copy_mesh_bundle(mesh_source, scene_dir, args.dry_run)
    for out_idx, source_image in enumerate(images):
        shutil.copy2(source_image, train_dir / f"{out_idx}.png")

    dump_json(scene_dir / "transforms_train.json", train_transforms)
    dump_json(scene_dir / "transforms_test.json", test_transforms)
    dump_json(scene_dir / "transforms_val.json", test_transforms)
    dump_json(scene_dir / "camera_poses.json", {"views": views})

    if args.copy_metadata:
        for name in ("args.json", "viewpoints.json"):
            source = texture_output / name
            if source.exists():
                shutil.copy2(source, scene_dir / name)

    assert_transforms_paths(scene_dir, train_transforms)

    manifest = {**plan, "copied_mesh_files": copied_mesh, "train_dir": str(train_dir)}
    dump_json(scene_dir / "prepare_gs_scene_manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = write_scene(args)
    print(json.dumps(manifest, indent=2))
    if args.dry_run:
        print("dry run only; no files were written")
    else:
        print(f"prepared GS scene at {manifest['scene_dir']} with {manifest['num_images']} training images")


if __name__ == "__main__":
    main()
