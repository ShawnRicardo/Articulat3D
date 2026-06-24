from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as guru
from tqdm import tqdm

from flow3d.init_utils import init_bg, init_fg_from_tracks_3d
from flow3d.loss_utils import (
    compute_accel_loss,
    compute_se3_smoothness_loss,
    compute_z_acc_loss,
    get_weights_for_procrustes,
    masked_l1_loss,
)
from flow3d.params import GaussianParams, MotionBases
from flow3d.tensor_dataclass import TrackObservations
from flow3d.transforms import solve_procrustes
from flow3d.vis.utils import project_2d_tracks


def compute_segmented_se3_smoothness_loss(
    rots: torch.Tensor,
    transls: torch.Tensor,
    segment_length: int,
) -> torch.Tensor:
    if segment_length <= 0 or rots.shape[1] <= segment_length:
        return compute_se3_smoothness_loss(rots, transls)

    losses = []
    for start in range(0, rots.shape[1], segment_length):
        end = min(start + segment_length, rots.shape[1])
        if end - start >= 3:
            losses.append(compute_se3_smoothness_loss(rots[:, start:end], transls[:, start:end]))
    if not losses:
        return rots.new_zeros(())
    return torch.stack(losses).mean()


def compute_segmented_accel_loss(values: torch.Tensor, segment_length: int) -> torch.Tensor:
    if segment_length <= 0 or values.shape[1] <= segment_length:
        return compute_accel_loss(values)

    losses = []
    for start in range(0, values.shape[1], segment_length):
        end = min(start + segment_length, values.shape[1])
        if end - start >= 3:
            losses.append(compute_accel_loss(values[:, start:end]))
    if not losses:
        return values.new_zeros(())
    return torch.stack(losses).mean()


def get_segment_neighbor_ts(num_frames: int, segment_length: int, device: torch.device) -> torch.Tensor:
    if segment_length <= 0:
        ts = torch.arange(1, max(num_frames - 1, 1), device=device)
    else:
        ts_parts = []
        for start in range(0, num_frames, segment_length):
            end = min(start + segment_length, num_frames)
            if end - start >= 3:
                ts_parts.append(torch.arange(start + 1, end - 1, device=device))
        if not ts_parts:
            return torch.empty(0, dtype=torch.long, device=device)
        ts = torch.cat(ts_parts)
    return ts.long()


def init_motion_params_with_mask_ids_articulat3d(
    tracks_3d: TrackObservations,
    mask_ids: torch.Tensor,
    num_bases: int | None,
    rot_type: Literal["6d"] = "6d",
    cano_t: int = 0,
    static_mask_id: int = 0,
    motion_segment_length: int = 200,
    min_cluster_points: int = 3,
    min_mean_weight: float = 0.05,
) -> tuple[MotionBases, torch.Tensor, TrackObservations]:
    if rot_type != "6d":
        raise ValueError("Articulat3D currently supports rot_type='6d' only")
    if mask_ids.shape[0] != tracks_3d.xyz.shape[0]:
        raise ValueError(
            f"mask_ids length {mask_ids.shape[0]} does not match tracks {tracks_3d.xyz.shape[0]}"
        )

    device = tracks_3d.xyz.device
    mask_ids = mask_ids.to(device=device, dtype=torch.long)
    num_frames = tracks_3d.xyz.shape[1]

    means_cano = tracks_3d.xyz[:, cano_t].clone()
    scene_center = means_cano.median(dim=0).values
    dists = torch.norm(means_cano - scene_center, dim=-1)
    valid_mask = (dists < torch.quantile(dists, 0.95)) & tracks_3d.visibles.any(dim=1)
    tracks_3d = tracks_3d.filter_valid(valid_mask)
    mask_ids = mask_ids[valid_mask]

    unique_ids = torch.unique(mask_ids).sort().values
    if num_bases is not None and num_bases != len(unique_ids):
        raise ValueError(
            f"num_motion_bases={num_bases} does not match unique mask_ids "
            f"{unique_ids.tolist()}"
        )
    num_bases_total = len(unique_ids)
    id_to_basis = {int(mid.item()): i for i, mid in enumerate(unique_ids)}
    guru.info(
        f"Initializing Articulat3D motion bases from mask_ids: "
        f"{[(int(i), int((mask_ids == i).sum().item())) for i in unique_ids]}"
    )

    motion_coefs = torch.full(
        (tracks_3d.xyz.shape[0], num_bases_total),
        -4.0,
        dtype=tracks_3d.xyz.dtype,
        device=device,
    )
    for mid in unique_ids:
        motion_coefs[mask_ids == mid, id_to_basis[int(mid.item())]] = 4.0

    id_rot = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device)
    init_rots = id_rot.reshape(1, 1, 6).repeat(num_bases_total, num_frames, 1)
    init_ts = torch.zeros(num_bases_total, num_frames, 3, device=device)
    errs_before = np.full((num_bases_total, num_frames), -1.0)
    errs_after = np.full((num_bases_total, num_frames), -1.0)

    tgt_ts = list(range(cano_t - 1, -1, -1)) + list(range(cano_t, num_frames))
    for mid in unique_ids:
        mid_int = int(mid.item())
        basis_idx = id_to_basis[mid_int]
        if mid_int == static_mask_id:
            continue

        cluster_mask = mask_ids == mid
        num_cluster_points = int(cluster_mask.sum().item())
        if num_cluster_points < min_cluster_points:
            guru.warning(
                f"Skipping mask_id={mid_int}: only {num_cluster_points} points, "
                f"need at least {min_cluster_points}"
            )
            continue

        cluster = tracks_3d.xyz[cluster_mask].transpose(0, 1)
        visibilities = tracks_3d.visibles[cluster_mask].swapaxes(0, 1)
        confidences = tracks_3d.confidences[cluster_mask].swapaxes(0, 1)
        weights = get_weights_for_procrustes(cluster, visibilities)

        prev_t = cano_t
        skipped = []
        for cur_t in tgt_ts:
            procrustes_weights = (
                weights[cano_t]
                * weights[cur_t]
                * (confidences[cano_t] + confidences[cur_t])
                / 2.0
            )
            if procrustes_weights.sum() < min_mean_weight * num_cluster_points:
                init_rots[basis_idx, cur_t] = init_rots[basis_idx, prev_t]
                init_ts[basis_idx, cur_t] = init_ts[basis_idx, prev_t]
                skipped.append(cur_t)
            else:
                se3, (err, err_before) = solve_procrustes(
                    src=cluster[cano_t],
                    dst=cluster[cur_t],
                    weights=procrustes_weights,
                    enforce_se3=True,
                    rot_type=rot_type,
                )
                init_rot, init_t, _ = se3
                init_rots[basis_idx, cur_t] = init_rot
                init_ts[basis_idx, cur_t] = init_t
                errs_after[basis_idx, cur_t] = err
                errs_before[basis_idx, cur_t] = err_before
            prev_t = cur_t
        if skipped:
            guru.info(f"mask_id={mid_int} reused previous transform for {len(skipped)} frames")

    if motion_segment_length > 0 and num_frames % motion_segment_length == 0:
        for start in range(motion_segment_length, num_frames, motion_segment_length):
            end = start + motion_segment_length
            init_rots[:, start:end] = init_rots[:, :motion_segment_length]
            init_ts[:, start:end] = init_ts[:, :motion_segment_length]

    guru.info(
        f"Articulat3D Procrustes error before={errs_before[errs_before >= 0].mean():.6f}, "
        f"after={errs_after[errs_after >= 0].mean():.6f}"
    )
    return MotionBases(init_rots, init_ts), motion_coefs, tracks_3d


def run_initial_optim_articulat3d(
    fg: GaussianParams,
    bases: MotionBases,
    tracks_3d: TrackObservations,
    Ks: torch.Tensor,
    w2cs: torch.Tensor,
    num_iters: int = 1000,
    use_depth_range_loss: bool = False,
    motion_segment_length: int = 200,
):
    optimizer = torch.optim.Adam(
        [
            {"params": bases.params["rots"], "lr": 1e-2},
            {"params": bases.params["transls"], "lr": 3e-2},
            {"params": fg.params["motion_coefs"], "lr": 1e-2},
            {"params": fg.params["means"], "lr": 1e-3},
        ],
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.1 ** (1 / num_iters)
    )
    num_frames = bases.num_frames
    if Ks.shape[0] != num_frames or w2cs.shape[0] != num_frames:
        raise ValueError(
            f"Initial optimization expects {num_frames} dynamic cameras, got "
            f"Ks={Ks.shape[0]}, w2cs={w2cs.shape[0]}"
        )
    device = bases.params["rots"].device
    w_smooth_func = lambda i, min_v, max_v, th: (
        min_v if i <= th else (max_v - min_v) * (i - th) / (num_iters - th) + min_v
    )

    gt_2d, gt_depth = project_2d_tracks(
        tracks_3d.xyz.swapaxes(0, 1), Ks, w2cs, return_depth=True
    )
    gt_2d = gt_2d.swapaxes(0, 1)
    gt_depth = gt_depth.swapaxes(0, 1)

    ts = torch.arange(0, num_frames, device=device)
    ts_mid = get_segment_neighbor_ts(num_frames, motion_segment_length, device)
    ts_neighbors = torch.cat((ts_mid - 1, ts_mid, ts_mid + 1)) if ts_mid.numel() else ts_mid

    pbar = tqdm(range(0, num_iters), desc="Articulat3D init optim")
    for i in pbar:
        coefs = fg.get_coefs()
        transfms = bases.compute_transforms(ts, coefs)
        positions = torch.einsum(
            "pnij,pj->pni",
            transfms,
            F.pad(fg.params["means"], (0, 1), value=1.0),
        )

        loss = 0.0
        track_3d_loss = masked_l1_loss(
            positions,
            tracks_3d.xyz,
            (tracks_3d.visibles.float() * tracks_3d.confidences)[..., None],
        )
        loss += track_3d_loss

        pred_2d, pred_depth = project_2d_tracks(
            positions.swapaxes(0, 1), Ks, w2cs, return_depth=True
        )
        pred_2d = pred_2d.swapaxes(0, 1)
        pred_depth = pred_depth.swapaxes(0, 1)
        loss_2d = (
            masked_l1_loss(
                pred_2d,
                gt_2d,
                (tracks_3d.invisibles.float() * tracks_3d.confidences)[..., None],
                quantile=0.95,
            )
            / Ks[0, 0, 0]
        )
        loss += 0.5 * loss_2d

        if use_depth_range_loss:
            near_depths = torch.quantile(gt_depth, 0.0, dim=0, keepdim=True)
            far_depths = torch.quantile(gt_depth, 0.98, dim=0, keepdim=True)
            loss_depth_in_range = 0
            if (pred_depth < near_depths).any():
                loss_depth_in_range += (near_depths - pred_depth)[pred_depth < near_depths].mean()
            if (pred_depth > far_depths).any():
                loss_depth_in_range += (pred_depth - far_depths)[pred_depth > far_depths].mean()
            loss += loss_depth_in_range * w_smooth_func(i, 0.05, 0.5, 400)

        motion_coef_sparse_loss = 1 - (coefs**2).sum(dim=-1).mean()
        loss += motion_coef_sparse_loss * 0.01

        w_smooth = w_smooth_func(i, 0.01, 0.1, 400)
        small_acc_loss = compute_segmented_se3_smoothness_loss(
            bases.params["rots"], bases.params["transls"], motion_segment_length
        )
        loss += small_acc_loss * w_smooth

        small_acc_loss_tracks = compute_segmented_accel_loss(positions, motion_segment_length)
        loss += small_acc_loss_tracks * w_smooth * 0.5

        if ts_mid.numel() > 0:
            transfms_nbs = bases.compute_transforms(ts_neighbors, coefs)
            means_nbs = torch.einsum(
                "pnij,pj->pni",
                transfms_nbs,
                F.pad(fg.params["means"], (0, 1), value=1.0),
            )
            means_nbs = means_nbs.reshape(means_nbs.shape[0], 3, -1, 3)
            z_accel_loss = compute_z_acc_loss(means_nbs, w2cs[ts_mid])
            loss += z_accel_loss * 0.1
        else:
            z_accel_loss = torch.zeros((), device=device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        pbar.set_description(
            f"init loss={loss.item():.3f} track3d={track_3d_loss.item():.3f} "
            f"smooth={small_acc_loss.item():.3f} z={z_accel_loss.item():.3f}"
        )


__all__ = [
    "compute_segmented_accel_loss",
    "compute_segmented_se3_smoothness_loss",
    "get_segment_neighbor_ts",
    "init_bg",
    "init_fg_from_tracks_3d",
    "init_motion_params_with_mask_ids_articulat3d",
    "run_initial_optim_articulat3d",
]
