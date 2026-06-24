# Copyright (c) TAPIP3D team(https://tapip3d.github.io/)

from concurrent.futures import ThreadPoolExecutor
import shlex
import tap
import torch
from typing import Optional, Tuple
from pathlib import Path
from datetime import datetime
from einops import repeat
from utils.common_utils import setup_logger
import logging
from annotation.megasam import MegaSAMAnnotator
import numpy as np
import cv2
from datasets.data_ops import _filter_one_depth
import torch.nn.functional as F

from utils.inference_utils import load_model, read_video, inference, get_grid_queries, resize_depth_bilinear

logger = logging.getLogger(__name__)

DEFAULT_QUERY_GRID_SIZE = 256
NUM_POINTS = 8192
MULTI_TRACK = 1

class Arguments(tap.Tap):
    input_path: str = "demo_inputs/table_31249.npz"
    device: str = "cuda"
    num_iters: int = 6
    support_grid_size: int = 16
    num_threads: int = 8
    resolution_factor: int = 1
    vis_threshold: Optional[float] = 0.9
    checkpoint: str = "checkpoints/tapip3d_final.pth"
    output_dir: str = "outputs/inference"
    depth_model: str = "moge"
    n_query_frames: int = 4
    n_query_points: int = 8192


def prepare_inputs(args, input_path: str, inference_res: Tuple[int, int], support_grid_size: int, num_threads: int = 8, device: str = "cpu"):
    if not Path (input_path).is_file():
        raise ValueError(f"Input file not found: {input_path}")
    video, depths, intrinsics, extrinsics, query_point, fg_mask = None, None, None, None, None, None
    if input_path.endswith((".mp4", ".avi", ".mov", ".webm")):
        video = read_video(input_path)
    elif input_path.endswith(".npz"):
        data = np.load(input_path)
        video = data['video']
        assert video.ndim == 4, f"Invalid video shape or dtype: {video.shape}"
        if video.dtype != np.uint8:
            video = (video * 255).astype(np.uint8)
        if video.shape[-1] != 3:
            video = np.transpose(video, (0, 2, 3, 1))
        depths = data.get('depths', None)
        intrinsics = data.get('intrinsics', None)
        extrinsics = data.get('extrinsics', None)
        query_point = data.get('query_point', None)
        fg_mask = data.get('fg_mask', None)
    else:
        raise ValueError(f"Unsupported input type: {input_path}. Supported formats are .mp4 and .npz.")
    
    assert depths is not None, "Depths must be provided in the input .npz file"
    _original_res = depths.shape[1:3]

    if intrinsics is None:
        raise ValueError("Intrinsics must be provided if depth is provided")
    if extrinsics is None:
        logger.info(f"No extrinsics provided, using identity matrix for all frames")
        extrinsics = repeat(np.eye(4), "i j -> t i j", t=len(video))
    
    intrinsics[:, 0, :] *= (inference_res[1] - 1) / (_original_res[1] - 1)
    intrinsics[:, 1, :] *= (inference_res[0] - 1) / (_original_res[0] - 1)

    # resize & remove edges
    with ThreadPoolExecutor(num_threads) as executor:
        video_futures = [executor.submit(cv2.resize, rgb, (inference_res[1], inference_res[0]), interpolation=cv2.INTER_LINEAR) for rgb in video]
        depths_futures = [executor.submit(resize_depth_bilinear, depth, (inference_res[1], inference_res[0])) for depth in depths]
        
        video = np.stack([future.result() for future in video_futures])
        depths = np.stack([future.result() for future in depths_futures])

        depths_futures = [executor.submit(_filter_one_depth, depth, 0.08, 15, intrinsic) for depth, intrinsic in zip(depths, intrinsics)]
        depths = np.stack([future.result() for future in depths_futures])

    video = (torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0).to(device)
    depths = torch.from_numpy(depths).float().to(device)
    intrinsics = torch.from_numpy(intrinsics).float().to(device)
    extrinsics = torch.from_numpy(extrinsics).float().to(device)
    if fg_mask is not None: # [T, H, W]
        fg_mask = torch.from_numpy(fg_mask).float().to(device)
        fg_mask = F.interpolate(fg_mask[:, None], (inference_res[0], inference_res[1]), 
                                mode='nearest')[:, 0] > 0.5
        depths[~fg_mask] = 0

    if query_point is None:
        support_grid_size = 0
        query_point = get_grid_queries(grid_size=DEFAULT_QUERY_GRID_SIZE, num_points=args.n_query_points, depths=depths, intrinsics=intrinsics, extrinsics=extrinsics)
        logger.info(f"No queries provided, using a grid at the first frame as queries")
    else:
        query_point = torch.from_numpy(query_point).float().to(device)

    return video, depths, intrinsics, extrinsics, query_point, support_grid_size


if __name__ == "__main__":
    setup_logger()
    args = Arguments().parse_args()

    output_dir = Path(args.output_dir)
    # output_dir = output_dir/ f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.checkpoint)
    model.to(args.device)

    inference_res = (int(model.image_size[0] * np.sqrt(args.resolution_factor)), int(model.image_size[1] * np.sqrt(args.resolution_factor)))
    print("inference_res", inference_res)
    model.set_image_size(inference_res)

    # Prepare inputs
    n_query_frames = args.n_query_frames
    args.n_query_points = args.n_query_points // n_query_frames
    video, depths, intrinsics, extrinsics, query_point, support_grid_size = prepare_inputs(
        args=args,
        input_path=args.input_path, 
        inference_res=inference_res, 
        support_grid_size=args.support_grid_size,
        num_threads=args.num_threads,
        device=args.device
    )

    # Run inference
    with torch.autocast("cuda", dtype=torch.bfloat16):
        coords, visibs = inference(
                model=model,
                video=video,
                depths=depths,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                query_point=query_point,
                num_iters=args.num_iters,
                grid_size=support_grid_size,
            )
        if n_query_frames > 1:
            all_coords, all_visibs = [coords], [visibs]
            video1 = video.flip(dims=(0,))
            depths1 = depths.flip(dims=(0,))
            intrinsics1 = intrinsics.flip(dims=(0,))
            extrinsics1 = extrinsics.flip(dims=(0,))
            query_point1 = get_grid_queries(grid_size=DEFAULT_QUERY_GRID_SIZE, num_points=args.n_query_points, depths=depths1, intrinsics=intrinsics1, extrinsics=extrinsics1)
            coords1, visibs1 = inference(
                model=model,
                video=video1,
                depths=depths1,
                intrinsics=intrinsics1,
                extrinsics=extrinsics1,
                query_point=query_point1,
                num_iters=args.num_iters,
                grid_size=support_grid_size,
            )
            all_coords.append(coords1.flip(dims=(0,)))
            all_visibs.append(visibs1.flip(dims=(0,)))
            if n_query_frames > 2:
                interval = len(video) // (n_query_frames - 1)
                for n in range(1, n_query_frames-1):
                    split_frame = n * interval
                    video1 = video[:split_frame+1].flip(dims=(0,))
                    depths1 = depths[:split_frame+1].flip(dims=(0,))
                    intrinsics1 = intrinsics[:split_frame+1].flip(dims=(0,))
                    extrinsics1 = extrinsics[:split_frame+1].flip(dims=(0,))
                    query_point1 = get_grid_queries(grid_size=DEFAULT_QUERY_GRID_SIZE, num_points=args.n_query_points, depths=depths1, intrinsics=intrinsics1, extrinsics=extrinsics1)
                    coords1, visibs1 = inference(
                        model=model,
                        video=video1,
                        depths=depths1,
                        intrinsics=intrinsics1,
                        extrinsics=extrinsics1,
                        query_point=query_point1,
                        num_iters=args.num_iters,
                        grid_size=support_grid_size,
                    )
                    video2 = video[split_frame:]    
                    depths2 = depths[split_frame:]
                    intrinsics2 = intrinsics[split_frame:]  
                    extrinsics2 = extrinsics[split_frame:]
                    coords2, visibs2 = inference(
                        model=model,
                        video=video2,
                        depths=depths2,
                        intrinsics=intrinsics2,
                        extrinsics=extrinsics2,
                        query_point=query_point1,
                        num_iters=args.num_iters,
                        grid_size=support_grid_size,
                    )
                    coords1 = coords1.flip(dims=(0,))
                    visibs1 = visibs1.flip(dims=(0,))
                    coords = torch.cat([coords1, coords2[1:]], dim=0)
                    visibs = torch.cat([visibs1, visibs2[1:]], dim=0)
                    all_coords.append(coords)
                    all_visibs.append(visibs)
            coords = torch.cat(all_coords, dim=1)
            visibs = torch.cat(all_visibs, dim=1)
    
    # Save results
    video = video.cpu().numpy()
    depths = depths.cpu().numpy()
    intrinsics = intrinsics.cpu().numpy()
    extrinsics = extrinsics.cpu().numpy()
    coords = coords.cpu().numpy()
    visibs = visibs.cpu().numpy()
    query_point = query_point.cpu().numpy()
    npz_path = Path(output_dir / Path(args.input_path).name).with_suffix(f".n{n_query_frames}.npz")
    npz_path.parent.mkdir(exist_ok=True, parents=True)
    np.savez(
        npz_path,
        video=video,
        depths=depths,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        coords=coords,
        visibs=visibs,
        query_points=query_point,
    )

    logger.info(f"Results saved to {npz_path.resolve()}.\nTo visualize them, run: `[bold yellow]python visualize.py {shlex.quote(str(npz_path.resolve()))}[/bold yellow]`")