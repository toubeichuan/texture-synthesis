#!/usr/bin/env python3
"""把 camera_poses.json 转成 NeRF/Blender 常用的 transforms.json 格式。

用途：
    export_camera_poses.py 会输出包含 views/c2w 的 camera_poses.json，本脚本
    将它转换为包含 camera_angle_x 和 frames 的 transforms JSON，方便后续渲染、
    训练或可视化流程读取。

使用方式：
    python scripts/camera/camera_poses_to_transforms.py \
        --input camera_poses.json \
        --out transforms.json

常用参数：
    --file-prefix ./train      输出 frame.file_path 的前缀
    --file-mode index          文件名后缀来源：index / view_idx / name
    --rotate-x-deg -90         对所有相机位姿额外施加世界 X 轴旋转
"""

import argparse
import json
import math
import os
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert camera_poses.json to transforms.json-style format."
    )
    parser.add_argument("--input", required=True, help="Path to camera_poses.json")
    parser.add_argument("--out", required=True, help="Output transforms json path, e.g. 1.json")
    parser.add_argument("--camera-angle-x", type=float, default=0.6911112070083618, help="camera_angle_x to store in output")
    parser.add_argument("--file-prefix", default="./train", help="Frame file_path prefix")
    parser.add_argument("--file-mode", choices=["index", "view_idx", "name"], default="index", help="How to generate frame file_path suffix")
    parser.add_argument("--rotate-x-deg", type=float, default=-90.0, help="Rotate every exported camera pose around world X axis")
    parser.add_argument("--rotate-y-deg", type=float, default=0.0, help="Rotate every exported camera pose around world Y axis")
    parser.add_argument("--rotate-z-deg", type=float, default=0.0, help="Rotate every exported camera pose around world Z axis")
    return parser.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_file_path(view: Dict[str, Any], index: int, prefix: str, mode: str) -> str:
    """根据指定模式生成 transforms.json 中每帧对应的图片路径。"""
    prefix = prefix.rstrip("/")
    if mode == "index":
        suffix = str(index)
    elif mode == "view_idx":
        suffix = str(view["view_idx"])
    else:
        raw_name = view.get("name", f"{index}")
        suffix = os.path.basename(str(raw_name)).replace("\\", "/")
    return f"{prefix}/{suffix}"


def rotation_x_matrix(angle_deg: float) -> List[List[float]]:
    angle = math.radians(angle_deg)
    c = math.cos(angle)
    s = math.sin(angle)
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, -s, 0.0],
        [0.0, s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rotation_y_matrix(angle_deg: float) -> List[List[float]]:
    angle = math.radians(angle_deg)
    c = math.cos(angle)
    s = math.sin(angle)
    return [
        [c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rotation_z_matrix(angle_deg: float) -> List[List[float]]:
    angle = math.radians(angle_deg)
    c = math.cos(angle)
    s = math.sin(angle)
    return [
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def matmul4(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j] = sum(a[i][k] * b[k][j] for k in range(4))
    return out


def build_pose_rotation(rx_deg: float, ry_deg: float, rz_deg: float) -> List[List[float]]:
    return matmul4(rotation_z_matrix(rz_deg), matmul4(rotation_y_matrix(ry_deg), rotation_x_matrix(rx_deg)))


def convert(
    data: Dict[str, Any],
    camera_angle_x: float,
    file_prefix: str,
    file_mode: str,
    pose_rotation: List[List[float]],
) -> Dict[str, Any]:
    """将输入 views 列表转换为 transforms.json 的 frames 列表。"""
    views: List[Dict[str, Any]] = data["views"]
    num_views = len(views)
    rotation = 2.0 * math.pi / num_views if num_views > 0 else 0.0

    frames = []
    for index, view in enumerate(views):
        transform_matrix = matmul4(pose_rotation, view["c2w"])
        frames.append(
            {
                "file_path": build_file_path(view, index, file_prefix, file_mode),
                "rotation": rotation,
                "transform_matrix": transform_matrix,
            }
        )

    return {
        "camera_angle_x": camera_angle_x,
        "frames": frames,
    }


def main() -> None:
    args = parse_args()
    data = load_json(args.input)
    pose_rotation = build_pose_rotation(args.rotate_x_deg, args.rotate_y_deg, args.rotate_z_deg)
    transforms = convert(data, args.camera_angle_x, args.file_prefix, args.file_mode, pose_rotation)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(transforms, f, indent=2)
    print(f"wrote {len(transforms['frames'])} frames to {args.out}")


if __name__ == "__main__":
    main()
