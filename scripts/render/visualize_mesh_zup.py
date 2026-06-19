#!/usr/bin/env python3
"""用 Matplotlib 可视化 Z-up 坐标系下的 mesh。

用途：
    快速检查 OBJ/PLY 等模型在原始 Z-up 坐标系中的朝向、尺度和包围盒。

使用方式：
    python scripts/render/visualize_mesh_zup.py --mesh model.obj --out preview.png
    python scripts/render/visualize_mesh_zup.py --mesh model.obj --show
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def load_mesh(mesh_path):
    """读取 mesh 文件，并把 Scene 合并成单个 Trimesh。"""
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError("trimesh is required to load mesh files") from exc

    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("failed to load mesh as Trimesh")
    return mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, required=True)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--show", action="store_true", help="show interactive window")
    args = parser.parse_args()

    mesh = load_mesh(args.mesh)
    verts = mesh.vertices
    faces = mesh.faces

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        triangles=faces,
        color=(0.7, 0.7, 0.7, 0.9),
        linewidth=0.2,
    )

    max_range = (verts.max(axis=0) - verts.min(axis=0)).max()
    mid = (verts.max(axis=0) + verts.min(axis=0)) / 2.0
    ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
    ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
    ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(os.path.basename(args.mesh))
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    if args.show:
        plt.show()
    if args.out:
        plt.savefig(args.out, dpi=200)
        print(f"saved visualization to {args.out}")


if __name__ == "__main__":
    main()
