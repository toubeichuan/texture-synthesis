#!/usr/bin/env python3
"""可视化 transforms.json 中的相机位置，可选叠加 OBJ 模型。

用途：
    检查 NeRF/Blender 风格 transforms.json 的相机中心、朝向和编号顺序。
    输出静态 Matplotlib 图片，适合快速排查相机矩阵是否翻转或错位。

使用方式：
    python scripts/camera/visualize_transforms.py --input transforms.json --obj cat.obj --out cameras.png
    python scripts/camera/visualize_transforms.py --input transforms.json --obj "" --show
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_frames(json_path):
    """读取 transforms.json 中的 frames 列表。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames", [])
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="transforms.json")
    parser.add_argument("--obj", type=str, default="cat.obj")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--show", action="store_true", help="show interactive window")
    parser.add_argument("--arrow_len", type=float, default=0.2)
    args = parser.parse_args()

    frames = load_frames(args.input)
    if not frames:
        raise ValueError(f"no frames in {args.input}")

    positions = []
    directions = []
    for frame in frames:
        mat = np.array(frame["transform_matrix"], dtype=np.float64)
        pos = mat[:3, 3]
        forward = mat[2, :3]
        positions.append(pos)
        directions.append(forward)

    positions = np.stack(positions, axis=0)
    directions = np.stack(directions, axis=0)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    mesh = None
    if args.obj:
        try:
            import trimesh
        except ImportError as exc:
            raise ImportError("trimesh is required to load the obj file") from exc
        mesh = trimesh.load(args.obj, force="mesh")

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    plot_x = positions[:, 0]
    plot_y = positions[:, 2]
    plot_z = positions[:, 1]
    ax.scatter(plot_x, plot_y, plot_z, s=20)
    ax.quiver(
        plot_x,
        plot_y,
        plot_z,
        directions[:, 0],
        directions[:, 2],
        directions[:, 1],
        length=args.arrow_len,
        normalize=True,
        linewidth=1,
    )
    for idx, (x, y, z) in enumerate(zip(plot_x, plot_y, plot_z)):
        ax.text(x, y, z, str(idx), fontsize=8)

    if mesh is not None:
        verts = mesh.vertices
        faces = mesh.faces
        ax.plot_trisurf(
            verts[:, 0],
            verts[:, 2],
            verts[:, 1],
            triangles=faces,
            color=(0.7, 0.7, 0.7, 0.35),
            linewidth=0.2,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.set_title(os.path.basename(args.input))
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    if args.show:
        plt.show()
    if args.out:
        plt.savefig(args.out, dpi=200)
        print(f"saved visualization to {args.out}")


if __name__ == "__main__":
    main()
