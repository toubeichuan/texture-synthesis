#!/usr/bin/env python3
"""用 viser 在浏览器中同时查看 mesh 和相机位姿。

用途：
    加载 OBJ/PLY/GLB 等 mesh，并叠加 camera_poses.json、transforms.json 或
    COLMAP sparse 文本模型中的相机视锥，检查视角方向、坐标轴和相机分布是否正确。

使用方式：
    python scripts/camera/visualize_camera_and_mesh.py --mesh easi-dyc/data/meshes/cat/cat.obj --poses camera_poses.json
    python scripts/camera/visualize_camera_and_mesh.py --mesh model.obj --transforms transforms.json --blender-to-yup
    python scripts/camera/visualize_camera_and_mesh.py --mesh model.obj --colmap path/to/sparse/0

打开地址：
    默认运行后访问 http://0.0.0.0:8080；如端口冲突可使用 --port 修改。
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import trimesh
import viser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a mesh together with camera poses exported by export_camera_poses.py."
    )
    parser.add_argument("--mesh", required=True, help="Path to mesh file (.obj/.ply/.glb/...)")
    parser.add_argument("--poses", default="", help="Path to camera_poses.json")
    parser.add_argument("--transforms", default="", help="Path to NeRF/Blender-style transforms_train.json")
    parser.add_argument("--colmap", default="", help="Path to COLMAP sparse model directory containing cameras.txt and images.txt")
    parser.add_argument("--colmap-images", default="", help="Path to COLMAP images.txt")
    parser.add_argument("--colmap-cameras", default="", help="Path to COLMAP cameras.txt")
    parser.add_argument("--host", default="0.0.0.0", help="Viewer host")
    parser.add_argument("--port", type=int, default=8080, help="Viewer port")
    parser.add_argument("--mesh-opacity", type=float, default=0.9, help="Mesh opacity")
    parser.add_argument("--mesh-color", type=int, nargs=3, default=[180, 180, 180], help="Fallback mesh RGB color")
    parser.add_argument("--frustum-scale", type=float, default=0.12, help="Camera frustum scale")
    parser.add_argument("--line-width", type=float, default=2.0, help="Frustum line width")
    parser.add_argument("--fov-deg", type=float, default=50.0, help="Vertical field of view in degrees")
    parser.add_argument("--aspect", type=float, default=1.0, help="Camera aspect ratio")
    parser.add_argument("--transforms-rotate-x-deg", type=float, default=0.0, help="Rotate imported --transforms camera poses around world X axis")
    parser.add_argument("--transforms-rotate-y-deg", type=float, default=0.0, help="Rotate imported --transforms camera poses around world Y axis")
    parser.add_argument("--transforms-rotate-z-deg", type=float, default=0.0, help="Rotate imported --transforms camera poses around world Z axis")

    parser.add_argument("--show-frames", action="store_true", help="Show a coordinate frame at each camera")
    parser.add_argument("--show-trajectory", action="store_true", help="Show a trajectory line through camera centers")
    parser.add_argument("--blender-to-yup", action="store_true", help="Apply a +90 deg world-X rotation to imported --transforms camera poses only")
    args = parser.parse_args()
    if not args.poses and not args.transforms and not args.colmap and not (args.colmap_images and args.colmap_cameras):
        parser.error("Provide one of --poses, --transforms, --colmap, or both --colmap-images and --colmap-cameras.")
    return args


def load_pose_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rotation_x_matrix(angle_deg: float) -> np.ndarray:
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


def rotation_y_matrix(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rotation_z_matrix(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def make_transform_rotation(args: argparse.Namespace) -> np.ndarray:
    rx = 90.0 if args.blender_to_yup else 0.0
    rx += args.transforms_rotate_x_deg
    ry = args.transforms_rotate_y_deg
    rz = args.transforms_rotate_z_deg
    return rotation_z_matrix(rz) @ rotation_y_matrix(ry) @ rotation_x_matrix(rx)


def load_transforms_views(path: Path, pose_fix: np.ndarray) -> List[Dict[str, Any]]:
    """读取 transforms.json，并统一整理成内部 views 结构。"""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    camera_angle_x = float(data.get("camera_angle_x", np.deg2rad(50.0)))
    views: List[Dict[str, Any]] = []
    for idx, frame in enumerate(data.get("frames", [])):
        c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
        c2w = pose_fix @ c2w
        views.append(
            {
                "view_idx": idx,
                "name": frame.get("file_path", f"frame_{idx}"),
                "selected_count": 0,
                "c2w": c2w.tolist(),
                "fov_x": camera_angle_x,
                "aspect": 1.0,
            }
        )
    return views


def qvec_to_rotmat(qvec: List[float]) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1.0 - 2.0 * y * y - 2.0 * z * z, 2.0 * x * y - 2.0 * w * z, 2.0 * x * z + 2.0 * w * y],
            [2.0 * x * y + 2.0 * w * z, 1.0 - 2.0 * x * x - 2.0 * z * z, 2.0 * y * z - 2.0 * w * x],
            [2.0 * x * z - 2.0 * w * y, 2.0 * y * z + 2.0 * w * x, 1.0 - 2.0 * x * x - 2.0 * y * y],
        ],
        dtype=np.float64,
    )


def parse_colmap_cameras(path: Path) -> Dict[int, Dict[str, Any]]:
    cameras: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            cameras[camera_id] = {
                "model": parts[1],
                "width": int(parts[2]),
                "height": int(parts[3]),
                "params": [float(v) for v in parts[4:]],
            }
    return cameras


def camera_intrinsics_to_fov_aspect(camera: Dict[str, Any]) -> Tuple[float, float]:
    width = float(camera["width"])
    height = float(camera["height"])
    params = camera["params"]
    model = camera["model"]

    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}:
        fy = params[0]
    elif model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
        fy = params[1]
    else:
        fy = params[0]

    fov_y = 2.0 * np.arctan(height / (2.0 * fy))
    aspect = width / height
    return float(fov_y), float(aspect)


def load_colmap_views(images_path: Path, cameras_path: Path) -> List[Dict[str, Any]]:
    cameras = parse_colmap_cameras(cameras_path)
    views: List[Dict[str, Any]] = []

    with images_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    for idx in range(0, len(lines), 2):
        parts = lines[idx].split()
        image_id = int(parts[0])
        qvec = [float(v) for v in parts[1:5]]
        tvec = np.asarray([float(v) for v in parts[5:8]], dtype=np.float64)
        camera_id = int(parts[8])
        name = parts[9]

        rotation_cw = qvec_to_rotmat(qvec)
        camera_center = -rotation_cw.T @ tvec
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = rotation_cw.T
        c2w[:3, 3] = camera_center

        fov, aspect = camera_intrinsics_to_fov_aspect(cameras[camera_id])
        views.append(
            {
                "view_idx": image_id,
                "name": name,
                "camera_id": camera_id,
                "selected_count": 0,
                "c2w": c2w.tolist(),
                "fov": fov,
                "aspect": aspect,
            }
        )

    return views


def to_numpy(mat_like: Any) -> np.ndarray:
    return np.asarray(mat_like, dtype=np.float64)


def rotation_matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    trace = np.trace(rotation)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rotation[2, 1] - rotation[1, 2]) * s
        y = (rotation[0, 2] - rotation[2, 0]) * s
        z = (rotation[1, 0] - rotation[0, 1]) * s
    else:
        if rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            w = (rotation[2, 1] - rotation[1, 2]) / s
            x = 0.25 * s
            y = (rotation[0, 1] + rotation[1, 0]) / s
            z = (rotation[0, 2] + rotation[2, 0]) / s
        elif rotation[1, 1] > rotation[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            w = (rotation[0, 2] - rotation[2, 0]) / s
            x = (rotation[0, 1] + rotation[1, 0]) / s
            y = 0.25 * s
            z = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            w = (rotation[1, 0] - rotation[0, 1]) / s
            x = (rotation[0, 2] + rotation[2, 0]) / s
            y = (rotation[1, 2] + rotation[2, 1]) / s
            z = 0.25 * s

    quat = np.asarray([w, x, y, z], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def easitex_c2w_to_viser_wxyz_position(c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # export_camera_poses.py 的相机本地轴为：+X 右、+Y 上、+Z 后方，也就是 -Z 朝前。
    # viser 的相机视锥接近 OpenCV 约定：+X 右、+Y 下、+Z 朝前，因此这里要做局部轴转换。
    convert_local = np.diag([1.0, -1.0, -1.0])
    rotation_world_from_cv = c2w[:3, :3] @ convert_local
    position = c2w[:3, 3]
    wxyz = rotation_matrix_to_wxyz(rotation_world_from_cv)
    return wxyz, position


def load_mesh(path: Path) -> Union[trimesh.Trimesh, trimesh.Scene]:
    mesh = trimesh.load(path, force="scene" if path.suffix.lower() in {".glb", ".gltf"} else None)
    if isinstance(mesh, trimesh.Scene) and len(mesh.geometry) == 1:
        return next(iter(mesh.geometry.values()))
    return mesh


def add_mesh(
    server: viser.ViserServer,
    mesh: Union[trimesh.Trimesh, trimesh.Scene],
    color: Tuple[int, int, int],
    opacity: float,
) -> None:
    def prepare_mesh(geom: trimesh.Trimesh) -> trimesh.Trimesh:
        geom = geom.copy()
        if not hasattr(geom.visual, "vertex_colors") or len(getattr(geom.visual, "vertex_colors", [])) == 0:
            rgba = np.array([color[0], color[1], color[2], int(max(0.0, min(1.0, opacity)) * 255)], dtype=np.uint8)
            geom.visual.vertex_colors = np.tile(rgba[None, :], (len(geom.vertices), 1))
        return geom

    if isinstance(mesh, trimesh.Scene):
        for idx, geom in enumerate(mesh.dump()):
            if not isinstance(geom, trimesh.Trimesh):
                continue
            geom = prepare_mesh(geom)
            server.scene.add_mesh_trimesh(
                f"/mesh/{idx}",
                mesh=geom,
            )
    else:
        mesh = prepare_mesh(mesh)
        server.scene.add_mesh_trimesh(
            "/mesh/main",
            mesh=mesh,
        )


def main() -> None:
    args = parse_args()
    mesh_path = Path(args.mesh)
    poses_path: Optional[Path] = Path(args.poses) if args.poses else None
    transforms_path: Optional[Path] = Path(args.transforms) if args.transforms else None
    transform_pose_fix = make_transform_rotation(args)

    if transforms_path is not None:
        views = load_transforms_views(transforms_path, transform_pose_fix)
        pose_source = transforms_path
    elif args.colmap:
        colmap_dir = Path(args.colmap)
        views = load_colmap_views(colmap_dir / "images.txt", colmap_dir / "cameras.txt")
        pose_source = colmap_dir
    elif args.colmap_images and args.colmap_cameras:
        views = load_colmap_views(Path(args.colmap_images), Path(args.colmap_cameras))
        pose_source = Path(args.colmap_images)
    else:
        assert poses_path is not None
        pose_data = load_pose_json(poses_path)
        views = pose_data["views"]
        pose_source = poses_path

    if len(views) == 0:
        raise ValueError("No camera views found in pose JSON.")

    mesh = load_mesh(mesh_path)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.add_grid("/grid", width=4.0, height=4.0)
    server.scene.add_frame("/world", axes_length=0.25, axes_radius=0.01)
    add_mesh(server, mesh, tuple(args.mesh_color), args.mesh_opacity)

    trajectory_points: list[np.ndarray] = []
    for view in views:
        c2w = to_numpy(view["c2w"])
        wxyz, position = easitex_c2w_to_viser_wxyz_position(c2w)
        camera_root = f"/cameras/cam_{view['view_idx']:03d}"
        if "fov" in view:
            fov = float(view["fov"])
        elif "fov_x" in view:
            # For square images, horizontal and vertical FOV are identical.
            fov = float(view["fov_x"])
        else:
            fov = float(np.deg2rad(args.fov_deg))
        aspect = float(view.get("aspect", args.aspect))

        server.scene.add_frame(
            camera_root,
            wxyz=wxyz,
            position=position,
            show_axes=False,
        )

        server.scene.add_camera_frustum(
            f"{camera_root}/frustum",
            fov=fov,
            aspect=aspect,
            scale=args.frustum_scale,
            line_width=args.line_width,
            color=(50, 120, 255) if view.get("selected_count", 0) == 0 else (255, 120, 40),
        )

        if args.show_frames:
            server.scene.add_frame(
                f"{camera_root}/axes",
                axes_length=max(args.frustum_scale * 0.8, 0.03),
                axes_radius=max(args.frustum_scale * 0.06, 0.003),
            )

        trajectory_points.append(position)

    if args.show_trajectory and len(trajectory_points) >= 2:
        pts = np.stack(trajectory_points, axis=0)
        segments = np.stack([pts[:-1], pts[1:]], axis=1)
        server.scene.add_line_segments(
            "/cameras/trajectory",
            points=segments,
            colors=np.tile(np.array([[255, 200, 0]], dtype=np.uint8), (segments.shape[0], 2, 1)),
            line_width=2.0,
        )

    print(f"Viewer running at http://{args.host}:{args.port}")
    print(f"Loaded mesh: {mesh_path}")
    print(f"Loaded poses: {pose_source}")

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
