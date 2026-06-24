import argparse
import json
import math
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image

root_path = os.path.abspath(__file__)
root_path = "/".join(root_path.split("/")[:-2])
sys.path.append(root_path)


DEFAULT_DEPTH_SCALE = 6553.5


def frame_name(frame_id):
    return f"{frame_id:06d}"


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def inclusive_range(start, end):
    if end < start:
        raise ValueError(f"Invalid frame range: {start}..{end}")
    return list(range(start, end + 1))


def load_intrinsics(scene_path):
    intrinsics_path = scene_path / "intrinsics.json"
    with open(intrinsics_path, "r") as f:
        intrinsics = np.array(json.load(f), dtype=np.float32)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"{intrinsics_path} must contain a 3x3 matrix, got {intrinsics.shape}")
    return intrinsics


def load_camera_poses(scene_path):
    camera_pose_path = scene_path / "camera_pose.json"
    with open(camera_pose_path, "r") as f:
        poses = json.load(f)
    if not isinstance(poses, dict):
        raise ValueError(f"{camera_pose_path} must be a dict keyed by frame id")
    return poses


def load_rgb_depth_mask(scene_path, frame_ids, depth_scale):
    rgbs, depths, masks, poses = [], [], [], []
    camera_poses = load_camera_poses(scene_path)
    for frame_id in frame_ids:
        name = frame_name(frame_id)
        image_path = scene_path / "images" / f"{name}.png"
        depth_path = scene_path / "depth" / f"{name}.png"
        mask_path = scene_path / "masks" / f"{name}.png"
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        if not depth_path.exists():
            raise FileNotFoundError(depth_path)
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        if name not in camera_poses:
            raise KeyError(f"Missing camera pose for frame {name}")

        image = np.array(Image.open(image_path))
        rgb = image[..., :3]
        alpha = image[..., 3] > 0 if image.ndim == 3 and image.shape[-1] == 4 else None
        depth = np.array(Image.open(depth_path)).astype(np.float32) / depth_scale
        mask = np.array(Image.open(mask_path)) > 0
        if alpha is not None:
            mask = mask & alpha
        depth[~mask] = 0

        rgbs.append(rgb)
        depths.append(depth)
        masks.append(mask.astype(np.float32))
        poses.append(np.array(camera_poses[name], dtype=np.float32))

    return np.stack(rgbs, 0), np.stack(depths, 0), np.stack(masks, 0), np.stack(poses, 0)


def write_ascii_ply(path, xyz, rgb, normals=None):
    if normals is None:
        normals = np.zeros_like(xyz, dtype=np.float32)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        data = np.concatenate([xyz.astype(np.float32), normals.astype(np.float32), rgb], axis=1)
        np.savetxt(f, data, fmt="%.6f %.6f %.6f %.6f %.6f %.6f %d %d %d")


def write_binary_ply(path, xyz, rgb, normals=None):
    if normals is None:
        normals = np.zeros_like(xyz, dtype=np.float32)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    vertices = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("nx", "<f4"),
            ("ny", "<f4"),
            ("nz", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"] = xyz[:, 0].astype(np.float32)
    vertices["y"] = xyz[:, 1].astype(np.float32)
    vertices["z"] = xyz[:, 2].astype(np.float32)
    vertices["nx"] = normals[:, 0].astype(np.float32)
    vertices["ny"] = normals[:, 1].astype(np.float32)
    vertices["nz"] = normals[:, 2].astype(np.float32)
    vertices["red"] = rgb[:, 0]
    vertices["green"] = rgb[:, 1]
    vertices["blue"] = rgb[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {xyz.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        vertices.tofile(f)


def voxel_downsample_first(xyz, rgb, voxel_size):
    if voxel_size <= 0 or xyz.shape[0] == 0:
        return xyz, rgb
    voxel = np.floor(xyz / voxel_size).astype(np.int64)
    _, keep = np.unique(voxel, axis=0, return_index=True)
    keep = np.sort(keep)
    return xyz[keep], rgb[keep]


def voxel_downsample_first_cuda(xyz, rgb, voxel_size):
    if voxel_size <= 0 or xyz.shape[0] == 0:
        return xyz, rgb
    import torch

    voxel = torch.floor(xyz / voxel_size).to(torch.int64)
    _, inverse = torch.unique(voxel, dim=0, return_inverse=True)
    first = torch.full((int(inverse.max().item()) + 1,), xyz.shape[0], dtype=torch.long, device=xyz.device)
    indices = torch.arange(xyz.shape[0], dtype=torch.long, device=xyz.device)
    first.scatter_reduce_(0, inverse, indices, reduce="amin", include_self=True)
    first = torch.sort(first).values
    return xyz[first], rgb[first]


def generate_cuda_point_cloud(scene_path, static_ids, intrinsics, depth_scale, pixel_stride, voxel_size, max_points):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment")

    device = torch.device("cuda")
    camera_poses = load_camera_poses(scene_path)
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    cv_to_gl = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float32, device=device))
    xyz_chunks, rgb_chunks = [], []

    for frame_id in static_ids:
        name = frame_name(frame_id)
        image = np.array(Image.open(scene_path / "images" / f"{name}.png"))
        depth_np = np.array(Image.open(scene_path / "depth" / f"{name}.png")).astype(np.float32) / depth_scale
        mask_np = np.array(Image.open(scene_path / "masks" / f"{name}.png")) > 0

        rgb = torch.from_numpy(image[..., :3].copy()).to(device=device, dtype=torch.uint8)
        depth = torch.from_numpy(depth_np).to(device=device, dtype=torch.float32)
        mask = torch.from_numpy(mask_np).to(device=device, dtype=torch.bool)
        if image.ndim == 3 and image.shape[-1] == 4:
            alpha = torch.from_numpy((image[..., 3] > 0).copy()).to(device=device, dtype=torch.bool)
            mask = mask & alpha

        depth = depth[::pixel_stride, ::pixel_stride]
        mask = mask[::pixel_stride, ::pixel_stride]
        rgb = rgb[::pixel_stride, ::pixel_stride]
        height, width = depth.shape
        ys = torch.arange(0, height * pixel_stride, pixel_stride, dtype=torch.float32, device=device)
        xs = torch.arange(0, width * pixel_stride, pixel_stride, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        valid = (depth > 0.01) & mask
        if not valid.any():
            continue
        z = depth[valid]
        x = (xx[valid] - cx) * z / fx
        y = (yy[valid] - cy) * z / fy
        pts_cv = torch.stack([x, y, z], dim=-1)
        pts_gl = pts_cv @ cv_to_gl.T
        pose = torch.tensor(camera_poses[name], dtype=torch.float32, device=device)
        pts_world = pts_gl @ pose[:3, :3].T + pose[:3, 3]

        xyz_chunks.append(pts_world)
        rgb_chunks.append(rgb[valid])

    if not xyz_chunks:
        raise ValueError("No valid points were generated from static frames")

    xyz = torch.cat(xyz_chunks, dim=0)
    rgb = torch.cat(rgb_chunks, dim=0)
    print(f"Generated {xyz.shape[0]} raw static-frame points on CUDA")
    xyz, rgb = voxel_downsample_first_cuda(xyz, rgb, voxel_size)
    print(f"Kept {xyz.shape[0]} points after CUDA voxel downsampling")
    if max_points > 0 and xyz.shape[0] > max_points:
        generator = torch.Generator(device=device)
        generator.manual_seed(0)
        keep = torch.randperm(xyz.shape[0], device=device, generator=generator)[:max_points]
        keep = torch.sort(keep).values
        xyz, rgb = xyz[keep], rgb[keep]
        print(f"Sampled {xyz.shape[0]} points after max-point cap")

    return xyz.detach().cpu().numpy(), rgb.detach().cpu().numpy()


def generate_cpu_point_cloud(scene_path, static_ids, intrinsics, depth_scale, pixel_stride, voxel_size, max_points):
    camera_poses = load_camera_poses(scene_path)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    cv_to_gl = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    xyz_all, rgb_all = [], []
    for frame_id in static_ids:
        name = frame_name(frame_id)
        image = np.array(Image.open(scene_path / "images" / f"{name}.png"))
        rgb = image[..., :3]
        alpha = image[..., 3] > 0 if image.ndim == 3 and image.shape[-1] == 4 else None
        depth = np.array(Image.open(scene_path / "depth" / f"{name}.png")).astype(np.float32) / depth_scale
        mask = np.array(Image.open(scene_path / "masks" / f"{name}.png")) > 0
        valid = (depth > 0.01) & mask
        if alpha is not None:
            valid = valid & alpha
        if pixel_stride > 1:
            stride_mask = np.zeros_like(valid, dtype=bool)
            stride_mask[::pixel_stride, ::pixel_stride] = True
            valid = valid & stride_mask
        ys, xs = np.nonzero(valid)
        if len(xs) == 0:
            continue
        z = depth[ys, xs]
        x = (xs.astype(np.float32) - cx) * z / fx
        y = (ys.astype(np.float32) - cy) * z / fy
        pts_cv = np.stack([x, y, z], axis=1)
        pts_gl = pts_cv @ cv_to_gl.T
        pose = np.array(camera_poses[name], dtype=np.float32)
        pts_world = pts_gl @ pose[:3, :3].T + pose[:3, 3]
        xyz_all.append(pts_world)
        rgb_all.append(rgb[ys, xs])

    if not xyz_all:
        raise ValueError("No valid points were generated from static frames")
    xyz = np.concatenate(xyz_all, axis=0)
    rgb = np.concatenate(rgb_all, axis=0)
    xyz, rgb = voxel_downsample_first(xyz, rgb, voxel_size)
    if max_points > 0 and xyz.shape[0] > max_points:
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(xyz.shape[0], size=max_points, replace=False))
        xyz, rgb = xyz[keep], rgb[keep]
    return xyz, rgb


def infer_image_size(scene_path):
    image_paths = sorted(glob(str(scene_path / "images" / "*.png")))
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found under {scene_path / 'images'}")
    with Image.open(image_paths[0]) as image:
        return image.size


def validate_frames(scene_path, frame_ids):
    camera_poses = load_camera_poses(scene_path)
    for frame_id in frame_ids:
        name = frame_name(frame_id)
        for folder in ["images", "depth", "masks"]:
            path = scene_path / folder / f"{name}.png"
            if not path.exists():
                raise FileNotFoundError(path)
        if name not in camera_poses:
            raise KeyError(f"Missing camera pose for frame {name}")


def write_transforms(scene_path, static_ids, dynamic_ids, depth_scale, overwrite):
    out_path = scene_path / "transforms.json"
    if out_path.exists() and not overwrite:
        print(f"{out_path} exists, skip. Use --reprocess to overwrite.")
        return

    intrinsics = load_intrinsics(scene_path)
    width, height = infer_image_size(scene_path)
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    camera_poses = load_camera_poses(scene_path)
    transforms = {
        "camera_angle_x": focal2fov(fx, width),
        "camera_angle_y": focal2fov(fy, height),
        "focal_x": fx,
        "focal_y": fy,
        "cx": cx,
        "cy": cy,
        "w": width,
        "h": height,
        "depth_scale": depth_scale,
        "frames": [],
    }

    for frame_id in static_ids:
        name = frame_name(frame_id)
        transforms["frames"].append({
            "file_path": f"images/{name}.png",
            "state": 0,
            "time": 0.0,
            "transform_matrix": camera_poses[name],
        })

    n_dynamic = max(1, len(dynamic_ids))
    for idx, frame_id in enumerate(dynamic_ids):
        name = frame_name(frame_id)
        transforms["frames"].append({
            "file_path": f"images/{name}.png",
            "state": 1,
            "time": idx / n_dynamic,
            "transform_matrix": camera_poses[name],
        })

    with open(out_path, "w") as f:
        json.dump(transforms, f, indent=4)
    print(f"Wrote {out_path} with {len(transforms['frames'])} frames")


def mobility_type_to_joint_prior_type(mobility_type):
    if mobility_type == "revolute":
        return "hinge"
    if mobility_type == "prismatic":
        return "slider"
    if mobility_type == "static":
        return "fixed"
    return mobility_type


def write_joint_priori(scene_path, overwrite):
    out_path = scene_path / "joint_priori.json"
    if out_path.exists() and not overwrite:
        print(f"{out_path} exists, skip. Use --reprocess to overwrite.")
        return

    mobility_path = scene_path / "gt" / "mobility_v2.json"
    if not mobility_path.exists():
        print(f"{mobility_path} not found, skip joint_priori.json generation.")
        return

    with open(mobility_path, "r") as f:
        mobility_infos = json.load(f)
    joint_priori_entries = []
    for info in mobility_infos:
        joint_priori_entries.append({
            "id": int(info.get("joint_id", len(joint_priori_entries))),
            "name": info.get("name", f"joint_{len(joint_priori_entries)}"),
            "joint": mobility_type_to_joint_prior_type(info.get("type", "fixed")),
            "parent": int(info.get("parent", -1)),
        })

    with open(out_path, "w") as f:
        json.dump(joint_priori_entries, f, indent=4)
    print(f"Wrote {out_path}")


def write_point_cloud(scene_path, static_ids, depth_scale, eps, cluster, visualize, overwrite, backend, pixel_stride, max_points):
    out_path = scene_path / "point_cloud.ply"
    if out_path.exists() and not overwrite:
        print(f"{out_path} exists, skip. Use --reprocess to overwrite.")
        return
    intrinsics = load_intrinsics(scene_path)
    if backend == "cuda":
        if cluster:
            print("CUDA point cloud backend ignores --cluster; voxel downsampling is still applied.")
        voxel_size = eps / 5.0
        xyz, rgb = generate_cuda_point_cloud(
            scene_path,
            static_ids,
            intrinsics,
            depth_scale,
            pixel_stride,
            voxel_size,
            max_points,
        )
        print(f"Saving CUDA-generated binary PLY to {out_path} with {xyz.shape[0]} points")
        write_binary_ply(out_path, xyz, rgb)
    elif backend == "cpu":
        if cluster:
            print("CPU point cloud backend ignores --cluster; voxel downsampling is still applied.")
        voxel_size = eps / 5.0
        xyz, rgb = generate_cpu_point_cloud(
            scene_path,
            static_ids,
            intrinsics,
            depth_scale,
            pixel_stride,
            voxel_size,
            max_points,
        )
        print(f"Saving point cloud to {out_path} with {xyz.shape[0]} points")
        write_binary_ply(out_path, xyz, rgb)
    else:
        raise ValueError(f"Unknown point cloud backend: {backend}")


def parse_args():
    parser = argparse.ArgumentParser("Prepare Articulat3DSimNew scene for Articulat3D/TAPIP3D")
    parser.add_argument("--scene_path", type=Path, required=True)
    parser.add_argument("--static_start", type=int, default=600)
    parser.add_argument("--static_end", type=int, default=749)
    parser.add_argument("--dynamic_start", type=int, default=0)
    parser.add_argument("--dynamic_end", type=int, default=599)
    parser.add_argument("--depth_scale", type=float, default=DEFAULT_DEPTH_SCALE)
    parser.add_argument("--point_cloud_eps", type=float, default=0.04)
    parser.add_argument("--point_cloud_backend", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--point_cloud_pixel_stride", type=int, default=4)
    parser.add_argument("--point_cloud_max_points", type=int, default=500000)
    parser.add_argument("--cluster", action="store_true", default=True)
    parser.add_argument("--no_cluster", action="store_false", dest="cluster")
    parser.add_argument("--skip_point_cloud", default=True)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--reprocess", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    scene_path = args.scene_path.resolve()
    static_ids = inclusive_range(args.static_start, args.static_end)
    dynamic_ids = inclusive_range(args.dynamic_start, args.dynamic_end)
    validate_frames(scene_path, static_ids + dynamic_ids)

    write_joint_priori(scene_path, args.reprocess)
    if not args.skip_point_cloud:
        write_point_cloud(
            scene_path,
            static_ids,
            args.depth_scale,
            args.point_cloud_eps,
            args.cluster,
            args.visualize,
            args.reprocess,
            args.point_cloud_backend,
            args.point_cloud_pixel_stride,
            args.point_cloud_max_points,
        )


if __name__ == "__main__":
    """
    python data_tools/process_data_vlmjson.py \
        --scene_path data/Articulat3DSimECCV/StorageFurniture_45194 \
        --skip_point_cloud \
        --reprocess
    """
    main()
