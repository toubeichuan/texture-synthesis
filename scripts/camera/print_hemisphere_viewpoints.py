#!/usr/bin/env python3
"""生成并打印半球/预设相机视角对应的 transforms.json。

用途：
    快速查看某组视角的相机位置和 transform_matrix，输出格式可直接作为
    transforms.json 使用。适合调试相机分布、验证坐标轴方向。

使用方式：
    python scripts/camera/print_hemisphere_viewpoints.py --num_viewpoints 10 --out transforms.json
    python scripts/camera/print_hemisphere_viewpoints.py --viewpoint_mode predefined --preset objaverse --out transforms.json

注意：
    输出 file_path 默认为 ./train/r_i；如果图像文件名不同，需要在后续流程中调整。
"""

import argparse
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
for candidate_name in ("easi-dyc", "easi", "easi-tex"):
    candidate_root = os.path.join(PROJECT_ROOT, candidate_name)
    if os.path.isdir(os.path.join(candidate_root, "lib")):
        sys.path.insert(0, candidate_root)
        break
else:
    raise ImportError("Could not find an EASI-style project folder containing lib/constants.py")

from lib.camera_helper import init_hemisphere_viewpoints
from lib.constants import VIEWPOINTS


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def camera_position_from_spherical_angles(dist, elev_deg, azim_deg):
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    x = dist * np.cos(elev) * np.sin(azim)
    y = dist * np.sin(elev)
    z = dist * np.cos(elev) * np.cos(azim)
    return np.array([x, y, z], dtype=np.float64)


def look_at_matrix(camera_pos, target=np.array([0.0, 0.0, 0.0])):
    """根据相机位置和目标点构造 camera-to-world 矩阵。"""
    up_world = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # 相机 forward 是 -Z，所以这里先算出相机的 “backward(+Z)” 方向。
    z = normalize(camera_pos - target)             # +Z: from target -> camera (backward)
    x = normalize(np.cross(up_world, z))           # +X: right
    y = np.cross(z, x)                             # +Y: up

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = camera_pos
    return c2w


def rotation_x_matrix(angle_deg):
    angle = np.deg2rad(angle_deg)
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_viewpoints", type=int, default=10)
    parser.add_argument(
        "--viewpoint_mode",
        type=str,
        default="hemisphere",
        choices=["hemisphere", "predefined"],
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="",
        help='for predefined mode: number in VIEWPOINTS (e.g. "6") or "objaverse"/"shapenet"',
    )
    parser.add_argument("--dist", type=float, default=1.0)
    parser.add_argument("--camera_angle_x", type=float, default=0.6911112070083618)
    parser.add_argument("--out", type=str, default="transforms.json")
    args = parser.parse_args()

    if args.viewpoint_mode == "hemisphere":
        dist_list, elev_list, azim_list, _ = init_hemisphere_viewpoints(
            args.num_viewpoints, args.dist
        )
    else:
        if not args.preset:
            raise ValueError("predefined mode requires --preset (e.g. 6 or objaverse)")
        preset_key = args.preset
        if preset_key.isdigit():
            preset_key = int(preset_key)
        if preset_key not in VIEWPOINTS:
            raise KeyError(f"preset {preset_key} not in VIEWPOINTS")
        elev_list = list(VIEWPOINTS[preset_key]["elev"])
        azim_list = list(VIEWPOINTS[preset_key]["azim"])
        dist_list = [args.dist for _ in elev_list]

    rotation = 2 * np.pi / len(elev_list)
    r_fix = rotation_x_matrix(90.0)
    frames = []
    for i, (dist, elev, azim) in enumerate(zip(dist_list, elev_list, azim_list)):
        cam_pos = camera_position_from_spherical_angles(dist, elev, azim)
        c2w = look_at_matrix(cam_pos)
        c2w = r_fix @ c2w
        frames.append(
            {
                "file_path": f"./train/r_{i}",
                "rotation": float(rotation),
                "transform_matrix": c2w.tolist(),
            }
        )

    data = {
        "camera_angle_x": float(args.camera_angle_x),
        "frames": frames,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"wrote {len(frames)} frames to {args.out}")


if __name__ == "__main__":
    main()
