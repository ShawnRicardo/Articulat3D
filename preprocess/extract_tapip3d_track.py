import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

root_path = os.path.abspath(__file__)
root_path = "/".join(root_path.split("/")[:-2])
sys.path.append(root_path)

import numpy as np
from PIL import Image
from tapip3d_motion_prior import build_tapip3d_motion_prior


DEFAULT_DEPTH_SCALE = 6553.5


def frame_name(frame_id):
    return f"{frame_id:06d}"


def inclusive_range(start, end):
    if end < start:
        raise ValueError(f"Invalid frame range: {start}..{end}")
    return list(range(start, end + 1))


def load_intrinsics(scene_path):
    with open(scene_path / "intrinsics.json", "r") as f:
        intrinsics = np.array(json.load(f), dtype=np.float32)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"{scene_path / 'intrinsics.json'} must contain a 3x3 matrix")
    return intrinsics


def load_camera_poses(scene_path):
    with open(scene_path / "camera_pose.json", "r") as f:
        camera_poses = json.load(f)
    if not isinstance(camera_poses, dict):
        raise ValueError(f"{scene_path / 'camera_pose.json'} must be keyed by frame id")
    return camera_poses

# 读取图片和相机参数
def load_dynamic_data(scene_path, frame_ids, depth_scale):
    camera_poses = load_camera_poses(scene_path)
    intrinsics = load_intrinsics(scene_path)
    video, depths, fg_masks, poses = [], [], [], []
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
        if image.ndim == 3 and image.shape[-1] == 4:
            alpha = image[..., 3:4].astype(np.float32) / 255.0
            rgb = image[..., :3].astype(np.float32) * alpha + (1.0 - alpha) * 255.0
            alpha_mask = alpha[..., 0] > 0.5
        else:
            rgb = image[..., :3].astype(np.float32)
            alpha_mask = np.ones(rgb.shape[:2], dtype=bool)
        mask = np.array(Image.open(mask_path)) > 0
        depth = np.array(Image.open(depth_path)).astype(np.float32) / depth_scale
        fg_mask = mask & alpha_mask
        depth[~fg_mask] = 0

        video.append(rgb.astype(np.uint8))
        depths.append(depth)
        fg_masks.append(fg_mask)
        poses.append(np.array(camera_poses[name], dtype=np.float32))

    video = np.stack(video, 0)
    depths = np.stack(depths, 0)
    fg_masks = np.stack(fg_masks, 0)
    poses = np.stack(poses, 0)
    poses[:, :3, :3] = poses[:, :3, :3] @ np.diag([1, -1, -1]).astype(np.float32)
    extrinsics = np.linalg.inv(poses)
    intrinsics = np.stack([intrinsics] * len(frame_ids), 0)
    return video, depths, fg_masks, intrinsics, extrinsics

# 把图像，相机参数等这些输入保存成 npz
def prepare_tapip3d_input(scene_path, scene_name, frame_ids, depth_scale, overwrite):
    input_path = scene_path / f"{scene_name}.npz"
    if input_path.exists() and not overwrite:
        print(f"{input_path} exists, skip input preparation. Use --reprocess to overwrite.")
        return input_path
    video, depths, fg_masks, intrinsics, extrinsics = load_dynamic_data(scene_path, frame_ids, depth_scale)
    print(
        "TAPIP3D input shapes:",
        f"video={video.shape}",
        f"depths={depths.shape}",
        f"fg_mask={fg_masks.shape}",
        f"intrinsics={intrinsics.shape}",
        f"extrinsics={extrinsics.shape}",
        f"depth_scale={depth_scale}",
    )
    np.savez(
        input_path,
        video=video,
        depths=depths,
        fg_mask=fg_masks,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
    )
    return input_path

# 调用 tapip3d
def run_tapip3d(tapip3d_dir, input_path, output_dir, n_query_frames, n_query_points):
    cmd = [
        sys.executable,
        "inference.py",
        "--input_path",
        str(input_path),
        "--n_query_frames",
        str(n_query_frames),
        "--n_query_points",
        str(n_query_points),
        "--output_dir",
        str(output_dir),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=tapip3d_dir, check=True)


def parse_args():
    parser = argparse.ArgumentParser("Run TAPIP3D on Articulat3DSimNew dynamic frames")
    parser.add_argument("--data_dir", type=Path, default=Path("data/Articulat3DSimECCV"))
    parser.add_argument("--scene_name", type=str, default="StorageFurniture_45194")
    parser.add_argument("--tapip3d_dir", type=Path, default=Path("TAPIP3D/"))
    parser.add_argument("--dynamic_start", type=int, default=0)
    parser.add_argument("--dynamic_end", type=int, default=199)
    parser.add_argument("--depth_scale", type=float, default=DEFAULT_DEPTH_SCALE)
    parser.add_argument("--base_query_frames", type=int, default=4)
    parser.add_argument("--n_query_points", type=int, default=8192)
    parser.add_argument("--reprocess", action="store_true")
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--keep_input", action="store_true")
    parser.add_argument("--skip_analysis", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--ignore_vis_mask", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir.resolve()
    scene_path = data_dir / args.scene_name
    tapip3d_dir = args.tapip3d_dir.resolve()
    if not scene_path.exists():
        raise FileNotFoundError(scene_path)
    if not tapip3d_dir.exists():
        raise FileNotFoundError(tapip3d_dir)

    joint_priori_path = scene_path / "joint_priori.json"
    with open(joint_priori_path, "r") as f:
        joint_priori_entries = json.load(f)
    n_query_frames = args.base_query_frames + len(joint_priori_entries) // 2
    output_path = scene_path / f"{args.scene_name}.n{n_query_frames}.npz"
    frame_ids = inclusive_range(args.dynamic_start, args.dynamic_end)

    input_path = prepare_tapip3d_input(
        scene_path,
        args.scene_name,
        frame_ids,
        args.depth_scale,
        args.reprocess,
    )
    if not output_path.exists() or args.reprocess:
        if args.skip_inference:
            raise FileNotFoundError(f"{output_path} does not exist and --skip_inference was set")
        run_tapip3d(tapip3d_dir, input_path, scene_path, n_query_frames, args.n_query_points)
    else:
        print(f"{output_path} exists, skip TAPIP3D inference. Use --reprocess to rerun.")

    if input_path.exists() and not args.keep_input:
        input_path.unlink()

    if args.skip_analysis:
        return

    results = build_tapip3d_motion_prior(
        args.scene_name,
        str(data_dir),
        n_query_frames,
        use_vis_mask=not args.ignore_vis_mask,
        visualize=args.visualize,
        print_info=True,
        realscan=False,
    )
    print("Joint analysis summary:")
    print(results)


if __name__ == "__main__":
    main()
