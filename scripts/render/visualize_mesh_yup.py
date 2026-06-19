#!/usr/bin/env python3
"""用 Matplotlib 可视化 Y-up 坐标系下的 mesh 和可选相机 transforms。

用途：
    快速生成一张静态 3D 预览图，检查 mesh 与 transforms.json 中相机位置是否对齐。
    绘图时会把世界 Y 轴作为竖直轴显示。

使用方式：
    python scripts/render/visualize_mesh_yup.py --mesh model.obj --out preview.png
    python scripts/render/visualize_mesh_yup.py --mesh model.obj --transforms transforms.json --show
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_mesh(mesh_path, max_faces=None):
    """读取 mesh；面数过多时尝试简化，避免 Matplotlib 绘制过慢。"""
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError("trimesh is required to load mesh files") from exc

    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("failed to load mesh as Trimesh")

    if max_faces is not None and len(mesh.faces) > max_faces:
        try:
            mesh = mesh.simplify_quadratic_decimation(max_faces)
        except Exception:
            pass

    return mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, default="")
    parser.add_argument("--transforms", type=str, default="")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--show", action="store_true", help="show interactive window")
    parser.add_argument("--arrow_len", type=float, default=0.2)
    parser.add_argument("--max_faces", type=int, default=20000)
    args = parser.parse_args()

    if not args.mesh and not args.transforms:
        raise ValueError("must provide --mesh or --transforms")

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    verts = None
    if args.mesh:
        mesh = load_mesh(args.mesh, max_faces=args.max_faces)
        verts = mesh.vertices
        faces = mesh.faces
        ax.plot_trisurf(
            verts[:, 0],
            verts[:, 2],
            verts[:, 1],
            triangles=faces,
            color=(0.7, 0.7, 0.7, 0.9),
            linewidth=0.2,
        )

    if args.transforms:
        with open(args.transforms, "r", encoding="utf-8") as f:
            data = json.load(f)
        frames = data.get("frames", [])
        positions = []
        directions = []
        for frame in frames:
            mat = np.array(frame["transform_matrix"], dtype=np.float64)
            pos = mat[:3, 3]
            forward = mat[2, :3]
            positions.append(pos)
            directions.append(forward)
        if positions:
            positions = np.stack(positions, axis=0)
            directions = np.stack(directions, axis=0)
            directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

            plot_x = positions[:, 0]
            plot_y = positions[:, 2]
            plot_z = positions[:, 1]
            ax.scatter(plot_x, plot_y, plot_z, s=20, color="red")
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
                color="red",
            )
            for idx, (x, y, z) in enumerate(zip(plot_x, plot_y, plot_z)):
                ax.text(x, y, z, str(idx), fontsize=8, color="red")
    
    
    if verts is not None:
        # scale = 3
        max_range = 5
        mid = (verts.max(axis=0) + verts.min(axis=0)) / 2.0
        ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
        ax.set_ylim(mid[2] - max_range / 2, mid[2] + max_range / 2)
        ax.set_zlim(mid[1] - max_range / 2, mid[1] + max_range / 2)
        ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.set_title(os.path.basename(args.mesh) if args.mesh else "camera_poses")
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    if args.show:
        plt.show()
    if args.out:
        plt.savefig(args.out, dpi=200)
        print(f"saved visualization to {args.out}")


if __name__ == "__main__":
    main()
