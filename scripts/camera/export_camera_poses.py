#!/usr/bin/env python3
"""导出相机位姿 camera_poses.json。

用途：
    从 viewpoints.json 或内置 VIEWPOINTS 预设生成相机位置，并计算每个视角的
    camera-to-world(c2w)、world-to-camera(w2c)、R、T 等矩阵。这个文件通常作为
    后续 transforms 转换、相机可视化、纹理生成验证的基础输入。

使用方式：
    1. 使用内置 8 视角预设：
        python scripts/camera/export_camera_poses.py --num_viewpoints 8 --dist 1.0 --out camera_poses.json

    2. 使用已有 viewpoints.json：
        python scripts/camera/export_camera_poses.py --viewpoints path/to/viewpoints.json --out camera_poses.json

    3. 使用指定预设：
        python scripts/camera/export_camera_poses.py --viewpoint_mode predefined --preset objaverse --out camera_poses.json

坐标约定：
    世界坐标为 +Y 朝上、+Z 朝前；相机本地坐标为 +X 右、+Y 上、-Z 看向目标。
"""

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple, Union


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
for candidate_name in ("easi-dyc", "easi", "easi-tex"):
    candidate_root = os.path.join(PROJECT_ROOT, candidate_name)
    if os.path.isdir(os.path.join(candidate_root, "lib")):
        sys.path.insert(0, candidate_root)
        break
else:
    raise ImportError("Could not find an EASI-style project folder containing lib/constants.py")

from lib.constants import VIEWPOINTS


def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def sub(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def norm(v: List[float]) -> float:
    return math.sqrt(dot(v, v))


def normalize(v: List[float]) -> List[float]:
    v_norm = norm(v)
    if v_norm == 0:
        return v
    return [x / v_norm for x in v]


def cross(a: List[float], b: List[float]) -> List[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def transpose_3x3(m: List[List[float]]) -> List[List[float]]:
    return [[m[j][i] for j in range(3)] for i in range(3)]


def matvec_3x3(m: List[List[float]], v: List[float]) -> List[float]:
    return [dot(row, v) for row in m]


def camera_position_from_spherical_angles(dist: float, elev_deg: float, azim_deg: float) -> List[float]:
    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)
    x = dist * math.cos(elev) * math.sin(azim)
    y = dist * math.sin(elev)
    z = dist * math.cos(elev) * math.cos(azim)
    return [x, y, z]


def look_at_c2w(camera_pos: List[float], target: List[float], up_world: List[float]) -> List[List[float]]:
    # 相机 forward 是 -Z，因此 c2w 的第三列保存相机本地 +Z，也就是从目标指回相机的方向。
    z = normalize(sub(camera_pos, target))
    x = normalize(cross(up_world, z))
    y = cross(z, x)

    return [
        [x[0], y[0], z[0], camera_pos[0]],
        [x[1], y[1], z[1], camera_pos[1]],
        [x[2], y[2], z[2], camera_pos[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def invert_rigid_c2w(c2w: List[List[float]]) -> List[List[float]]:
    rotation = [row[:3] for row in c2w[:3]]
    translation = [row[3] for row in c2w[:3]]
    rotation_t = transpose_3x3(rotation)
    inv_translation = [-v for v in matvec_3x3(rotation_t, translation)]
    return [
        rotation_t[0] + [inv_translation[0]],
        rotation_t[1] + [inv_translation[1]],
        rotation_t[2] + [inv_translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def load_viewpoints(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def init_predefined_viewpoints(sample_space: int, init_dist: float) -> Tuple[List[float], List[float], List[float], List[str]]:
    if sample_space not in VIEWPOINTS:
        available = sorted(str(k) for k in VIEWPOINTS.keys() if isinstance(k, int))
        raise KeyError(f"sample_space={sample_space} not in VIEWPOINTS; available numeric presets: {', '.join(available)}")

    viewpoints = VIEWPOINTS[sample_space]
    dist_list = [init_dist for _ in range(sample_space)]
    elev_list = [float(v) for v in viewpoints["elev"]]
    azim_list = [float(v) for v in viewpoints["azim"]]
    sector_list = [str(v) for v in viewpoints["sector"]]
    return dist_list, elev_list, azim_list, sector_list


def init_preset_viewpoints(preset: Union[int, str], init_dist: float) -> Tuple[List[float], List[float], List[float], List[str]]:
    if preset not in VIEWPOINTS:
        raise KeyError(f"preset={preset} not in VIEWPOINTS")

    viewpoints = VIEWPOINTS[preset]
    count = len(viewpoints["elev"])
    dist_list = [init_dist for _ in range(count)]
    elev_list = [float(v) for v in viewpoints["elev"]]
    azim_list = [float(v) for v in viewpoints["azim"]]
    sector_list = [str(v) for v in viewpoints["sector"]]
    return dist_list, elev_list, azim_list, sector_list


def init_hemisphere_viewpoints(sample_space: int, init_dist: float) -> Tuple[List[float], List[float], List[float], List[str]]:
    """使用 Fibonacci/golden-angle 采样生成上半球视角。"""
    num_points = 2 * sample_space
    if num_points < 2:
        raise ValueError("num_viewpoints must be >= 1")

    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    elev_list: List[float] = []
    azim_list: List[float] = []

    for i in range(num_points):
        y = 1.0 - (i / float(num_points - 1)) * 2.0
        if y < 0:
            continue

        theta = golden_angle * i
        elev_list.append(math.degrees(math.asin(y)))
        azim_list.append(math.degrees(theta))

    dist_list = [init_dist for _ in elev_list]
    sector_list = ["good" for _ in elev_list]
    return dist_list, elev_list, azim_list, sector_list


def generate_viewpoints(viewpoint_mode: str, num_viewpoints: int, dist: float, preset: str) -> Dict[str, Any]:
    if viewpoint_mode == "hemisphere":
        dist_list, elev_list, azim_list, sector_list = init_hemisphere_viewpoints(num_viewpoints, dist)
    elif viewpoint_mode == "predefined":
        if preset:
            preset_key: Union[int, str]
            preset_key = int(preset) if preset.isdigit() else preset
            dist_list, elev_list, azim_list, sector_list = init_preset_viewpoints(preset_key, dist)
        else:
            dist_list, elev_list, azim_list, sector_list = init_predefined_viewpoints(num_viewpoints, dist)
    else:
        raise ValueError(f"unsupported viewpoint_mode={viewpoint_mode}")

    return {
        "dist": dist_list,
        "elev": elev_list,
        "azim": azim_list,
        "sector": sector_list,
        "view": [],
    }


def build_pose_records(viewpoints: Dict[str, Any], target: List[float]) -> List[Dict[str, Any]]:
    """把 dist/elev/azim 视角参数转换成完整相机位姿记录。"""
    dist_list = viewpoints["dist"]
    elev_list = viewpoints["elev"]
    azim_list = viewpoints["azim"]
    sectors = viewpoints.get("sector", [""] * len(dist_list))
    selected = viewpoints.get("view", [])

    up_world = [0.0, 1.0, 0.0]
    records = []

    for idx, (dist, elev, azim, sector) in enumerate(zip(dist_list, elev_list, azim_list, sectors)):
        camera_pos = camera_position_from_spherical_angles(float(dist), float(elev), float(azim))
        c2w = look_at_c2w(camera_pos, target=target, up_world=up_world)
        w2c = invert_rigid_c2w(c2w)

        records.append(
            {
                "view_idx": idx,
                "sector": sector,
                "selected_count": selected.count(idx),
                "dist": float(dist),
                "elev": float(elev),
                "azim": float(azim),
                "camera_pos": camera_pos,
                "target": target,
                "up_world": up_world,
                "c2w": c2w,
                "w2c": w2c,
                "R": [row[:3] for row in w2c[:3]],
                "T": [row[3] for row in w2c[:3]],
            }
        )

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline export camera poses from viewpoints.json or from EASI-Tex-style viewpoint generation."
    )
    parser.add_argument("--viewpoints", default="", help="Path to viewpoints.json")
    parser.add_argument("--out", default="", help="Output JSON path")
    parser.add_argument("--target", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="Look-at target in world coordinates")
    parser.add_argument("--viewpoint_mode", choices=["predefined", "hemisphere"], default="predefined", help="Used when --viewpoints is not provided")
    parser.add_argument("--num_viewpoints", type=int, default=8, help="Used when --viewpoints is not provided")
    parser.add_argument("--dist", type=float, default=1.0, help="Used when --viewpoints is not provided")
    parser.add_argument("--preset", default="", help='Optional predefined preset, e.g. "6", "12", "20", "objaverse", "shapenet"')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = [float(v) for v in args.target]

    if args.viewpoints:
        viewpoints = load_viewpoints(args.viewpoints)
        source = os.path.abspath(args.viewpoints)
        default_out = os.path.join(os.path.dirname(args.viewpoints), "camera_poses.json")
    else:
        viewpoints = generate_viewpoints(args.viewpoint_mode, args.num_viewpoints, args.dist, args.preset)
        source = "generated"
        default_out = os.path.abspath("camera_poses.json")

    records = build_pose_records(viewpoints, target)
    out_path = os.path.abspath(args.out) if args.out else default_out

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_viewpoints": source,
                "generation_args": {
                    "viewpoint_mode": args.viewpoint_mode,
                    "num_viewpoints": args.num_viewpoints,
                    "dist": args.dist,
                    "preset": args.preset,
                } if not args.viewpoints else None,
                "coordinate_system": {
                    "up_axis": "+Y",
                    "front_axis": "+Z",
                    "camera_forward": "-Z",
                },
                "target": target,
                "num_views": len(records),
                "views": records,
            },
            f,
            indent=4,
        )

    print(f"wrote {len(records)} camera poses to {out_path}")


if __name__ == "__main__":
    main()
