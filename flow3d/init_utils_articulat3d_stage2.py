from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import roma
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger as guru

from flow3d.articulated_motion_articulat3d_stage2 import (
    JOINT_TYPE_TO_ID,
    PRISMATIC_ID,
    REVOLUTE_ID,
    STATIC_ID,
    ArticulatedMotionBasesArticulat3DStage2,
    SceneModelArticulat3DStage2,
    Stage2MotionReport,
)
from flow3d.loss_utils import knn as compute_knn
from flow3d.params import GaussianParams
from flow3d.scene_model import SceneModel
from flow3d.transforms import cont_6d_to_rmat


@dataclass
class JointCandidate:
    source_index: int
    joint_type: str
    joint_type_id: int
    axis: torch.Tensor
    pivot: torch.Tensor


@dataclass
class GTJointAxis:
    source_index: int
    joint_type: str
    axis: torch.Tensor
    pivot: torch.Tensor | None = None


def _axis_angle_from_rotmats(R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    trace = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]).clamp(-1.0, 3.0)
    theta = torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))
    sin_theta = torch.sin(theta)
    axis_x = (R[..., 2, 1] - R[..., 1, 2]) / (2 * sin_theta.clamp(min=1e-8))
    axis_y = (R[..., 0, 2] - R[..., 2, 0]) / (2 * sin_theta.clamp(min=1e-8))
    axis_z = (R[..., 1, 0] - R[..., 0, 1]) / (2 * sin_theta.clamp(min=1e-8))
    axis = torch.stack([axis_x, axis_y, axis_z], dim=-1)
    axis = F.normalize(axis, dim=-1, eps=1e-8)
    axis = torch.where(
        (sin_theta.abs() > 1e-6)[..., None],
        axis,
        axis.new_tensor([1.0, 0.0, 0.0]).expand_as(axis),
    )
    return theta, axis


def _fit_revolute_pivot(R_seq: torch.Tensor, t_seq: torch.Tensor) -> torch.Tensor:
    T = R_seq.shape[0]
    eye = torch.eye(3, device=R_seq.device, dtype=R_seq.dtype)
    A = eye[None].expand(T, 3, 3) - R_seq
    b = t_seq
    p, *_ = torch.linalg.lstsq(A.reshape(-1, 3), b.reshape(-1, 1))
    return p.squeeze(-1)


def _fit_revolute(R_seq: torch.Tensor, t_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    theta, axes = _axis_angle_from_rotmats(R_seq)
    valid = theta.abs() > 1e-5
    if valid.any():
        weighted_axis = (axes[valid] * theta[valid, None]).sum(dim=0)
        axis = F.normalize(weighted_axis, dim=0, eps=1e-8)
        if axis.norm() < 1e-6:
            axis = axes[valid][0]
    else:
        axis = R_seq.new_tensor([1.0, 0.0, 0.0])
    signed_theta = theta * torch.sign((axes * axis[None]).sum(dim=-1)).clamp(min=-1.0, max=1.0)
    signed_theta = torch.where(valid, signed_theta, signed_theta.new_zeros(()).expand_as(signed_theta))
    pivot = _fit_revolute_pivot(R_seq, t_seq)
    return axis, pivot, signed_theta


def _fit_prismatic(t_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centered = t_seq - t_seq.mean(dim=0, keepdim=True)
    if centered.norm() < 1e-8:
        axis = t_seq.new_tensor([1.0, 0.0, 0.0])
    else:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        axis = F.normalize(vh[0], dim=0, eps=1e-8)
        values = t_seq @ axis
        if values[-1] < values[0]:
            axis = -axis
    values = t_seq @ axis
    return axis, values


def _infer_joint_type(R_seq: torch.Tensor, t_seq: torch.Tensor) -> str:
    theta, _ = _axis_angle_from_rotmats(R_seq)
    angle_range = (theta.max() - theta.min()).item()
    trans_range = (t_seq.norm(dim=-1).max() - t_seq.norm(dim=-1).min()).item()
    if angle_range < 1e-3 and trans_range < 1e-4:
        return "static"
    return "revolute" if angle_range > trans_range else "prismatic"


def _parse_joint_type(info: dict) -> tuple[str, int] | None:
    raw = str(
        info.get("joint_type", info.get("type", info.get("joint", "fixed")))
    ).lower()
    if raw in {"s", "static", "fixed", "base"}:
        return "static", STATIC_ID
    if raw in {"r", "revolute", "hinge"}:
        return "revolute", REVOLUTE_ID
    if raw in {"p", "prismatic", "slider"}:
        return "prismatic", PRISMATIC_ID
    return None


def _normalization_tensors(
    scene_norm_transform: torch.Tensor | None,
    scene_norm_scale: torch.Tensor | float | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scene_norm_transform is None:
        transform = torch.eye(4, device=device, dtype=dtype)
    else:
        transform = scene_norm_transform.to(device=device, dtype=dtype)
    if scene_norm_scale is None:
        scale = torch.ones((), device=device, dtype=dtype)
    else:
        scale = torch.as_tensor(scene_norm_scale, device=device, dtype=dtype)
    return transform, scale


def _normalize_annotation_point(
    point: list[float] | tuple[float, ...],
    transform: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    p = torch.as_tensor(point, device=transform.device, dtype=transform.dtype)
    return (transform[:3, :3] @ p + transform[:3, 3]) / scale


def _normalize_annotation_dir(
    direction: list[float] | tuple[float, ...],
    transform: torch.Tensor,
) -> torch.Tensor:
    d = torch.as_tensor(direction, device=transform.device, dtype=transform.dtype)
    d = transform[:3, :3] @ d
    return F.normalize(d, dim=0, eps=1e-8)


def load_joint_candidates_from_annotation(
    scene_path: str | Path,
    device: torch.device,
    dtype: torch.dtype,
    scene_norm_transform: torch.Tensor | None = None,
    scene_norm_scale: torch.Tensor | float | None = None,
) -> list[JointCandidate]:
    path = Path(scene_path) / "joint_details.json"
    if not path.exists():
        return []
    with open(path, "r") as f:
        infos = json.load(f)
    if not isinstance(infos, list):
        return []
    transform, scale = _normalization_tensors(
        scene_norm_transform, scene_norm_scale, device, dtype
    )
    candidates = []
    for source_index, info in enumerate(infos):
        parsed = _parse_joint_type(info)
        if parsed is None:
            guru.warning(f"Unknown joint type in {path}: {info}")
            continue
        joint_type, joint_type_id = parsed
        if joint_type_id == STATIC_ID:
            continue
        direction = info.get("direction", info.get("axis_dir", None))
        if direction is None:
            guru.warning(f"Skipping joint candidate without direction: {info}")
            continue
        axis = _normalize_annotation_dir(direction, transform)
        if axis.norm() < 1e-6:
            guru.warning(f"Skipping joint candidate with zero direction: {info}")
            continue
        pivot_src = info.get("origin", info.get("axis_point", info.get("center", [0, 0, 0])))
        pivot = (
            _normalize_annotation_point(pivot_src, transform, scale)
            if joint_type_id == REVOLUTE_ID
            else torch.zeros(3, device=device, dtype=dtype)
        )
        candidates.append(
            JointCandidate(
                source_index=source_index,
                joint_type=joint_type,
                joint_type_id=joint_type_id,
                axis=axis,
                pivot=pivot,
            )
        )
    return candidates


def _make_axis_segments(
    anchors: torch.Tensor,
    axes: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    if anchors.numel() == 0:
        return torch.empty(0, 2, 3, device=anchors.device, dtype=anchors.dtype)
    axes = F.normalize(axes, dim=-1, eps=1e-8)
    half = 0.5 * lengths[:, None].clamp(min=1e-6)
    return torch.stack((anchors - axes * half, anchors + axes * half), dim=1)


def _match_gt_revolute_axes_to_bases(
    gt_axes: list[GTJointAxis],
    basis_indices: torch.Tensor,
    bases: ArticulatedMotionBasesArticulat3DStage2,
) -> list[GTJointAxis]:
    use_count = min(len(gt_axes), int(basis_indices.numel()))
    if use_count == 0:
        return []
    if len(gt_axes) != int(basis_indices.numel()) or len(gt_axes) > 8:
        return gt_axes[:use_count]

    basis_pivots = bases.params["pivot"][basis_indices].detach()
    basis_axes = bases.normalized_axes()[basis_indices].detach()
    best_perm = None
    best_cost = float("inf")
    for perm in itertools.permutations(range(len(gt_axes)), use_count):
        cost = 0.0
        for basis_row, gt_idx in enumerate(perm):
            gt = gt_axes[gt_idx]
            if gt.pivot is None:
                cost += 1e6
                continue
            pivot_cost = (gt.pivot - basis_pivots[basis_row]).norm().item()
            axis_cost = 1.0 - abs(float((gt.axis * basis_axes[basis_row]).sum().item()))
            cost += pivot_cost + 0.05 * axis_cost
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
    if best_perm is None:
        return gt_axes[:use_count]
    return [gt_axes[i] for i in best_perm]


def load_gt_joint_viz_from_mobility(
    scene_path: str | Path,
    bases: ArticulatedMotionBasesArticulat3DStage2,
    scene_norm_transform: torch.Tensor | None,
    scene_norm_scale: torch.Tensor | float | None,
    gt_joint_axis_path: str | Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Load GT joint axes only for viewer diagnostics.

    The returned tensors are visualization buffers. They must not be used for
    fitting, losses, assignment refinement, or optimizer logic.
    """

    device = bases.params["axis_raw"].device
    dtype = bases.params["axis_raw"].dtype
    if gt_joint_axis_path is None:
        path = Path(scene_path) / "gt" / "mobility_v2.json"
    else:
        path = Path(gt_joint_axis_path)

    empty_segments = torch.empty(0, 2, 3, device=device, dtype=dtype)
    empty_pivots = torch.empty(0, 3, device=device, dtype=dtype)
    report = {
        "path": str(path),
        "exists": bool(path.exists()),
        "used_for_training": False,
        "num_gt_prismatic": 0,
        "num_gt_revolute": 0,
        "num_used_prismatic": 0,
        "num_used_revolute": 0,
    }
    if not path.exists():
        return empty_segments, empty_segments, empty_pivots, report

    with open(path, "r") as f:
        infos = json.load(f)
    if not isinstance(infos, list):
        report["error"] = "mobility_v2.json is not a list"
        return empty_segments, empty_segments, empty_pivots, report

    transform, scale = _normalization_tensors(
        scene_norm_transform, scene_norm_scale, device, dtype
    )
    gt_prismatic: list[GTJointAxis] = []
    gt_revolute: list[GTJointAxis] = []
    for source_index, info in enumerate(infos):
        joint_type = str(info.get("type", info.get("joint_type", ""))).lower()
        if joint_type not in {"revolute", "prismatic"}:
            continue
        direction = info.get("axis_dir", info.get("direction", None))
        if direction is None:
            guru.warning(f"Skipping GT joint axis without direction in {path}: {info}")
            continue
        axis = _normalize_annotation_dir(direction, transform)
        if axis.norm() < 1e-6:
            guru.warning(f"Skipping GT joint axis with zero direction in {path}: {info}")
            continue
        if joint_type == "revolute":
            if "axis_point" not in info:
                guru.warning(f"Skipping GT revolute joint without axis_point in {path}: {info}")
                continue
            pivot = _normalize_annotation_point(info["axis_point"], transform, scale)
            gt_revolute.append(
                GTJointAxis(
                    source_index=source_index,
                    joint_type=joint_type,
                    axis=axis,
                    pivot=pivot,
                )
            )
        else:
            gt_prismatic.append(
                GTJointAxis(source_index=source_index, joint_type=joint_type, axis=axis)
            )

    report["num_gt_prismatic"] = len(gt_prismatic)
    report["num_gt_revolute"] = len(gt_revolute)

    prismatic_basis = torch.nonzero(
        bases.joint_type_ids == PRISMATIC_ID, as_tuple=False
    ).flatten()
    revolute_basis = torch.nonzero(
        bases.joint_type_ids == REVOLUTE_ID, as_tuple=False
    ).flatten()

    use_prismatic = min(len(gt_prismatic), int(prismatic_basis.numel()))
    if use_prismatic > 0:
        prismatic_basis = prismatic_basis[:use_prismatic]
        p_axes = torch.stack([gt_prismatic[i].axis for i in range(use_prismatic)], dim=0)
        p_anchors = bases.viz_anchor[prismatic_basis].detach()
        p_lengths = bases.viz_length[prismatic_basis].detach()
        gt_prismatic_segments = _make_axis_segments(p_anchors, p_axes, p_lengths)
    else:
        gt_prismatic_segments = empty_segments

    matched_revolute = _match_gt_revolute_axes_to_bases(gt_revolute, revolute_basis, bases)
    use_revolute = len(matched_revolute)
    if use_revolute > 0:
        revolute_basis = revolute_basis[:use_revolute]
        r_axes = torch.stack([gt.axis for gt in matched_revolute], dim=0)
        r_pivots = torch.stack([gt.pivot for gt in matched_revolute if gt.pivot is not None], dim=0)
        r_lengths = bases.viz_length[revolute_basis].detach()
        gt_revolute_segments = _make_axis_segments(r_pivots, r_axes, r_lengths)
        gt_revolute_pivots = r_pivots
    else:
        gt_revolute_segments = empty_segments
        gt_revolute_pivots = empty_pivots

    report["num_used_prismatic"] = int(use_prismatic)
    report["num_used_revolute"] = int(use_revolute)
    report["prismatic_source_indices"] = [
        int(gt_prismatic[i].source_index) for i in range(use_prismatic)
    ]
    report["revolute_source_indices"] = [
        int(gt.source_index) for gt in matched_revolute
    ]
    if len(gt_prismatic) != int(torch.sum(bases.joint_type_ids == PRISMATIC_ID).item()):
        report["prismatic_count_mismatch"] = True
    if len(gt_revolute) != int(torch.sum(bases.joint_type_ids == REVOLUTE_ID).item()):
        report["revolute_count_mismatch"] = True

    return gt_prismatic_segments, gt_revolute_segments, gt_revolute_pivots, report


def _trimmed_mean(values: torch.Tensor, trim_quantile: float) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(())
    if values.numel() < 3:
        return values.mean()
    med = values.median()
    keep = (values - med).abs() <= torch.quantile((values - med).abs(), trim_quantile)
    return values[keep].mean() if keep.any() else med


def _circular_trimmed_mean(angles: torch.Tensor, trim_quantile: float) -> torch.Tensor:
    if angles.numel() == 0:
        return angles.new_zeros(())
    if angles.numel() < 3:
        return torch.atan2(torch.sin(angles).mean(), torch.cos(angles).mean())
    med = angles.median()
    diff = torch.atan2(torch.sin(angles - med), torch.cos(angles - med)).abs()
    keep = diff <= torch.quantile(diff, trim_quantile)
    vals = angles[keep] if keep.any() else angles
    return torch.atan2(torch.sin(vals).mean(), torch.cos(vals).mean())


def _trajectory_residual_stats(
    pred: torch.Tensor, target: torch.Tensor, trim_quantile: float
) -> dict[str, float]:
    errors = (pred - target).norm(dim=-1).reshape(-1)
    if errors.numel() == 0:
        return {"cost": float("inf"), "median": float("inf"), "mean": float("inf"), "q90": float("inf"), "q98": float("inf")}
    cutoff = torch.quantile(errors, trim_quantile) if errors.numel() > 2 else errors.max()
    trimmed = errors[errors <= cutoff]
    return {
        "cost": float(trimmed.mean().item()),
        "median": float(errors.median().item()),
        "mean": float(errors.mean().item()),
        "q90": float(torch.quantile(errors, 0.90).item()),
        "q98": float(torch.quantile(errors, 0.98).item()),
    }


def _fit_prismatic_points(
    points: torch.Tensor,
    axis: torch.Tensor,
    trim_quantile: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    axis = F.normalize(axis, dim=0, eps=1e-8)
    disps = points - points[:, :1]
    values = []
    for t in range(points.shape[1]):
        values.append(_trimmed_mean(disps[:, t] @ axis, trim_quantile))
    values = torch.stack(values)
    pred = points[:, :1] + values[None, :, None] * axis[None, None]
    return values, pred


def _fit_revolute_points(
    points: torch.Tensor,
    axis: torch.Tensor,
    pivot: torch.Tensor,
    trim_quantile: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    axis = F.normalize(axis, dim=0, eps=1e-8)
    src = points[:, 0] - pivot
    src_plane = src - (src @ axis)[:, None] * axis[None]
    src_unit = F.normalize(src_plane, dim=-1, eps=1e-8)
    values = []
    for t in range(points.shape[1]):
        dst = points[:, t] - pivot
        dst_plane = dst - (dst @ axis)[:, None] * axis[None]
        dst_unit = F.normalize(dst_plane, dim=-1, eps=1e-8)
        sin = (torch.cross(src_unit, dst_unit, dim=-1) * axis[None]).sum(dim=-1)
        cos = (src_unit * dst_unit).sum(dim=-1).clamp(-1.0, 1.0)
        values.append(_circular_trimmed_mean(torch.atan2(sin, cos), trim_quantile))
    values = torch.stack(values)
    rot = torch.stack([roma.rotvec_to_rotmat(v * axis) for v in values], dim=0)
    transl = pivot[None] - torch.einsum("tij,j->ti", rot, pivot)
    pred = torch.einsum("tij,nj->nti", rot, points[:, 0]) + transl[None]
    return values, pred


def _part_viz_anchor_and_length(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    p0 = points[:, 0]
    anchor = p0.median(dim=0).values
    if p0.shape[0] < 2:
        return anchor, p0.new_tensor(0.25)
    lo = torch.quantile(p0, 0.05, dim=0)
    hi = torch.quantile(p0, 0.95, dim=0)
    length = (1.25 * (hi - lo).norm()).clamp(min=0.25, max=2.0)
    return anchor, length


def _select_fit_indices(
    labels: torch.Tensor,
    confidences: torch.Tensor,
    part_idx: int,
    min_conf: float,
    max_points: int,
    min_points: int,
) -> torch.Tensor:
    part_indices = torch.nonzero(labels == part_idx, as_tuple=False).squeeze(-1)
    if part_indices.numel() == 0:
        return part_indices
    part_conf = confidences[part_indices]
    cutoff = max(float(min_conf), float(part_conf.median().item()))
    fit_indices = part_indices[part_conf >= cutoff]
    min_points = min(min_points, part_indices.numel())
    if fit_indices.numel() < min_points:
        order = torch.argsort(part_conf, descending=True)
        fit_indices = part_indices[order[:min_points]]
    if fit_indices.numel() > max_points:
        fit_conf = confidences[fit_indices]
        order = torch.argsort(fit_conf, descending=True)
        fit_indices = fit_indices[order[:max_points]]
    return fit_indices


def _label_counts(labels: torch.Tensor, num_bases: int) -> list[int]:
    return torch.bincount(labels.detach().cpu(), minlength=num_bases).tolist()


def _robust_time_mean(errors: torch.Tensor, trim_quantile: float) -> torch.Tensor:
    if errors.shape[-1] == 0:
        return errors.new_zeros(errors.shape[:-1])
    if errors.shape[-1] < 3:
        return errors.mean(dim=-1)
    cutoff = torch.quantile(errors, trim_quantile, dim=-1, keepdim=True)
    keep = errors <= cutoff
    return (errors * keep).sum(dim=-1) / keep.sum(dim=-1).clamp_min(1)


@torch.no_grad()
def _compute_stage2_assignment_residuals(
    stage1: SceneModel,
    bases: ArticulatedMotionBasesArticulat3DStage2,
    canonical_means: torch.Tensor,
    frame_stride: int,
    trim_quantile: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = canonical_means.device
    num_frames = bases.num_frames
    stride = max(int(frame_stride), 1)
    ts = torch.arange(0, num_frames, stride, device=device)
    if ts.numel() == 0 or int(ts[-1].item()) != num_frames - 1:
        ts = torch.cat([ts, torch.tensor([num_frames - 1], device=device, dtype=ts.dtype)])

    base_rots, base_transls = bases.compute_base_transforms(ts)
    base_rots = base_rots.detach()
    base_transls = base_transls.detach()
    num_gaussians = canonical_means.shape[0]
    residuals = torch.empty(
        num_gaussians,
        bases.num_bases,
        device=device,
        dtype=canonical_means.dtype,
    )

    for start in range(0, num_gaussians, chunk_size):
        end = min(start + chunk_size, num_gaussians)
        inds = torch.arange(start, end, device=device)
        stage1_points, _ = stage1.compute_poses_fg(ts, inds=inds)
        stage1_points = stage1_points.detach()
        cano = canonical_means[inds]
        pred = torch.einsum("ktij,nj->nkti", base_rots, cano) + base_transls[None]
        errors = (pred - stage1_points[:, None]).norm(dim=-1)
        residuals[start:end] = _robust_time_mean(errors, trim_quantile)
    return residuals, ts


def _part_residual_stats(
    residuals: torch.Tensor,
    labels: torch.Tensor,
    current_residuals: torch.Tensor,
    bad_quantile: float,
) -> tuple[torch.Tensor, list[dict]]:
    num_bases = residuals.shape[1]
    thresholds = torch.full(
        (num_bases,),
        float("inf"),
        device=residuals.device,
        dtype=residuals.dtype,
    )
    stats = []
    for part_idx in range(num_bases):
        mask = labels == part_idx
        vals = current_residuals[mask]
        if vals.numel() == 0:
            stats.append(
                {
                    "part_idx": part_idx,
                    "count": 0,
                    "q50": float("nan"),
                    "q80": float("nan"),
                    "q90": float("nan"),
                    "bad_threshold": float("inf"),
                }
            )
            continue
        q50 = torch.quantile(vals, 0.50)
        q80 = torch.quantile(vals, 0.80)
        q90 = torch.quantile(vals, 0.90)
        threshold = torch.quantile(vals, bad_quantile)
        thresholds[part_idx] = threshold
        stats.append(
            {
                "part_idx": part_idx,
                "count": int(vals.numel()),
                "q50": float(q50.item()),
                "q80": float(q80.item()),
                "q90": float(q90.item()),
                "bad_threshold": float(threshold.item()),
            }
        )
    return thresholds, stats


def _build_knn_spatial_targets(
    canonical_means: torch.Tensor,
    labels: torch.Tensor,
    confidences: torch.Tensor,
    residuals: torch.Tensor,
    residual_bad_mask: torch.Tensor,
    assignment_train_conf_thresh: float,
    spatial_knn_k: int,
    spatial_majority_ratio: float,
    spatial_residual_slack_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    num_gaussians, num_bases = residuals.shape
    device = canonical_means.device
    spatial_targets = labels.clone()
    spatial_valid = torch.zeros(num_gaussians, dtype=torch.bool, device=device)
    spatial_train_mask = torch.zeros(num_gaussians, dtype=torch.bool, device=device)
    if num_gaussians <= 1 or spatial_knn_k <= 0:
        return spatial_targets, spatial_valid, spatial_train_mask, {
            "enabled": False,
            "reason": "not_enough_points_or_zero_k",
        }

    k = min(int(spatial_knn_k), num_gaussians - 1)
    distances_np, indices_np = compute_knn(canonical_means.detach(), k)
    distances = torch.as_tensor(distances_np, device=device, dtype=canonical_means.dtype)
    indices = torch.as_tensor(indices_np, device=device, dtype=torch.long)

    if distances.numel() > 0:
        global_cutoff = torch.quantile(distances.reshape(-1), 0.95).clamp_min(1e-6)
        local_cutoff = (distances.median(dim=1).values[:, None] * 3.0).clamp_min(
            global_cutoff * 0.1
        )
        distance_valid = distances <= torch.minimum(local_cutoff, global_cutoff)
    else:
        distance_valid = torch.ones_like(indices, dtype=torch.bool)

    trusted = (confidences >= assignment_train_conf_thresh) & (~residual_bad_mask)
    neighbor_labels = labels[indices]
    neighbor_valid = trusted[indices] & distance_valid
    neighbor_counts = neighbor_valid.sum(dim=1)
    one_hot = F.one_hot(neighbor_labels.clamp(min=0), num_classes=num_bases).to(
        dtype=canonical_means.dtype
    )
    counts = (one_hot * neighbor_valid[..., None]).sum(dim=1)
    majority_counts, majority_labels = counts.max(dim=-1)
    majority_ratio = majority_counts / neighbor_counts.clamp_min(1).to(majority_counts.dtype)

    arange = torch.arange(num_gaussians, device=device)
    current_residuals = residuals[arange, labels]
    majority_residuals = residuals[arange, majority_labels]
    residual_ok = majority_residuals <= (
        current_residuals * (1.0 + spatial_residual_slack_ratio) + 1e-8
    )
    spatial_valid = (
        (neighbor_counts > 0)
        & (majority_ratio >= spatial_majority_ratio)
        & residual_ok
    )
    spatial_targets[spatial_valid] = majority_labels[spatial_valid]
    spatial_train_mask = spatial_valid & (spatial_targets != labels)
    stats = {
        "enabled": True,
        "k": k,
        "majority_ratio": float(spatial_majority_ratio),
        "residual_slack_ratio": float(spatial_residual_slack_ratio),
        "spatial_valid_count": int(spatial_valid.sum().item()),
        "spatial_trainable_count": int(spatial_train_mask.sum().item()),
        "spatial_switch_count": int(spatial_train_mask.sum().item()),
        "mean_trusted_neighbor_count": float(neighbor_counts.float().mean().item()),
        "mean_majority_ratio": float(majority_ratio[neighbor_counts > 0].mean().item())
        if (neighbor_counts > 0).any()
        else 0.0,
    }
    return spatial_targets, spatial_valid, spatial_train_mask, stats


@torch.no_grad()
def _build_assignment_graph_from_stage1(
    stage1: SceneModel,
    canonical_means: torch.Tensor,
    spatial_knn_k: int,
    graph_frame_stride: int,
    graph_motion_quantile: float,
    chunk_size: int,
    edge_chunk_size: int = 500_000,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    num_gaussians = canonical_means.shape[0]
    device = canonical_means.device
    dtype = canonical_means.dtype
    if num_gaussians <= 1 or spatial_knn_k <= 0:
        return (
            torch.empty(0, 2, dtype=torch.long, device=device),
            torch.empty(0, dtype=dtype, device=device),
            {"enabled": False, "reason": "not_enough_points_or_zero_k"},
        )

    k = min(int(spatial_knn_k), num_gaussians - 1)
    distances_np, indices_np = compute_knn(canonical_means.detach(), k)
    distances = torch.as_tensor(distances_np, device=device, dtype=dtype)
    indices = torch.as_tensor(indices_np, device=device, dtype=torch.long)
    global_cutoff = torch.quantile(distances.reshape(-1), 0.95).clamp_min(1e-6)
    local_cutoff = (distances.median(dim=1).values[:, None] * 3.0).clamp_min(
        global_cutoff * 0.1
    )
    spatial_valid = distances <= torch.minimum(local_cutoff, global_cutoff)

    stride = max(int(graph_frame_stride), 1)
    ts = torch.arange(0, stage1.num_frames, stride, device=device)
    if ts.numel() == 0 or int(ts[-1].item()) != stage1.num_frames - 1:
        ts = torch.cat([ts, torch.tensor([stage1.num_frames - 1], device=device)])
    feature_dim = int(ts.numel() * 3)
    motion_features = torch.empty(num_gaussians, feature_dim, device=device, dtype=dtype)
    for start in range(0, num_gaussians, chunk_size):
        end = min(start + chunk_size, num_gaussians)
        inds = torch.arange(start, end, device=device)
        points, _ = stage1.compute_poses_fg(ts, inds=inds)
        disp = points - points[:, :1]
        motion_features[start:end] = disp.reshape(end - start, -1)

    src = torch.arange(num_gaussians, device=device)[:, None].expand(-1, k).reshape(-1)
    dst = indices.reshape(-1)
    spatial_valid_flat = spatial_valid.reshape(-1)
    distances_flat = distances.reshape(-1)
    motion_dist = torch.empty_like(distances_flat)
    for start in range(0, motion_dist.numel(), edge_chunk_size):
        end = min(start + edge_chunk_size, motion_dist.numel())
        diff = motion_features[src[start:end]] - motion_features[dst[start:end]]
        motion_dist[start:end] = diff.norm(dim=-1) / math.sqrt(max(feature_dim, 1))

    if spatial_valid_flat.any():
        motion_threshold = torch.quantile(
            motion_dist[spatial_valid_flat],
            float(graph_motion_quantile),
        ).clamp_min(1e-8)
    else:
        motion_threshold = motion_dist.new_tensor(float("inf"))
    keep = spatial_valid_flat & (motion_dist <= motion_threshold)
    src_keep = src[keep]
    dst_keep = dst[keep]
    spatial_weight = torch.exp(-distances_flat[keep] / global_cutoff.clamp_min(1e-8))
    motion_weight = torch.exp(-motion_dist[keep] / motion_threshold.clamp_min(1e-8))
    weights = (spatial_weight * motion_weight).clamp_min(1e-6)
    edges = torch.stack([src_keep, dst_keep], dim=-1).long()
    stats = {
        "enabled": True,
        "k": k,
        "frame_stride": int(stride),
        "sampled_frame_count": int(ts.numel()),
        "edge_count_before_filter": int(src.numel()),
        "spatial_edge_count": int(spatial_valid_flat.sum().item()),
        "edge_count": int(edges.shape[0]),
        "mean_neighbor_count": float(edges.shape[0] / max(num_gaussians, 1)),
        "motion_quantile": float(graph_motion_quantile),
        "motion_threshold": float(motion_threshold.item()),
        "global_spatial_cutoff": float(global_cutoff.item()),
    }
    return edges, weights, stats


@torch.no_grad()
def refine_assignment_targets_from_stage1_residuals(
    stage1: SceneModel,
    bases: ArticulatedMotionBasesArticulat3DStage2,
    labels: torch.Tensor,
    confidences: torch.Tensor,
    assignment_train_mask: torch.Tensor,
    assignment_train_conf_thresh: float,
    enable_assignment_refine: bool,
    assignment_refine_stride: int,
    assignment_residual_bad_quantile: float,
    assignment_switch_improve_ratio: float,
    enable_assignment_knn_prior: bool,
    spatial_knn_k: int,
    spatial_majority_ratio: float,
    spatial_residual_slack_ratio: float,
    trim_quantile: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    num_gaussians = labels.shape[0]
    num_bases = bases.num_bases
    target_labels = labels.detach().clone()
    spatial_target_labels = labels.detach().clone()
    spatial_valid_mask = torch.zeros(num_gaussians, dtype=torch.bool, device=labels.device)
    train_mask = assignment_train_mask.detach().clone()
    report = {
        "enabled": bool(enable_assignment_refine),
        "initial_label_counts": _label_counts(labels, num_bases),
        "initial_trainable_count": int(train_mask.sum().item()),
        "final_trainable_count": int(train_mask.sum().item()),
        "spatial_valid_count": 0,
    }
    if not enable_assignment_refine:
        return train_mask, target_labels, spatial_target_labels, spatial_valid_mask, report

    cano_ts = torch.zeros(1, dtype=torch.long, device=labels.device)
    canonical_means, _ = stage1.compute_poses_fg(cano_ts)
    canonical_means = canonical_means[:, 0].detach()
    residuals, sampled_ts = _compute_stage2_assignment_residuals(
        stage1=stage1,
        bases=bases,
        canonical_means=canonical_means,
        frame_stride=assignment_refine_stride,
        trim_quantile=trim_quantile,
        chunk_size=chunk_size,
    )

    arange = torch.arange(num_gaussians, device=labels.device)
    current_residuals = residuals[arange, labels]
    best_residuals, best_labels = residuals.min(dim=-1)
    if num_bases > 1:
        two_best = torch.topk(residuals, k=2, dim=-1, largest=False).values
        margin_small = (two_best[:, 1] - two_best[:, 0]) <= (
            current_residuals.clamp_min(1e-8) * assignment_switch_improve_ratio
        )
    else:
        margin_small = torch.zeros_like(train_mask)

    thresholds, part_stats = _part_residual_stats(
        residuals,
        labels,
        current_residuals,
        assignment_residual_bad_quantile,
    )
    residual_bad = current_residuals > thresholds[labels]
    residual_switch = (best_labels != labels) & (
        best_residuals <= current_residuals * (1.0 - assignment_switch_improve_ratio)
    )
    confidence_train_mask = assignment_train_mask
    train_mask = train_mask | residual_bad | margin_small | residual_switch

    knn_stats = {"enabled": False}
    knn_train_mask = torch.zeros_like(train_mask)
    if enable_assignment_knn_prior:
        spatial_target_labels, spatial_valid_mask, knn_train_mask, knn_stats = (
            _build_knn_spatial_targets(
                canonical_means=canonical_means,
                labels=labels,
                confidences=confidences,
                residuals=residuals,
                residual_bad_mask=residual_bad,
                assignment_train_conf_thresh=assignment_train_conf_thresh,
                spatial_knn_k=spatial_knn_k,
                spatial_majority_ratio=spatial_majority_ratio,
                spatial_residual_slack_ratio=spatial_residual_slack_ratio,
            )
        )
        # Residual switches are stronger evidence than local smoothing; avoid conflicting priors.
        conflict = residual_switch & spatial_valid_mask & (spatial_target_labels != target_labels)
        spatial_valid_mask[conflict] = False
        train_mask = train_mask | knn_train_mask

    diagnostic_spatial_valid_count = int(spatial_valid_mask.sum().item())
    target_labels = labels.detach().clone()
    spatial_target_labels = labels.detach().clone()
    spatial_valid_mask = torch.zeros_like(spatial_valid_mask)

    report = {
        "enabled": True,
        "sampled_frame_stride": int(max(assignment_refine_stride, 1)),
        "sampled_frame_count": int(sampled_ts.numel()),
        "sampled_frame_first": int(sampled_ts[0].item()) if sampled_ts.numel() > 0 else None,
        "sampled_frame_last": int(sampled_ts[-1].item()) if sampled_ts.numel() > 0 else None,
        "chunk_size": int(chunk_size),
        "initial_label_counts": _label_counts(labels, num_bases),
        "best_residual_label_counts": _label_counts(best_labels, num_bases),
        "initial_trainable_count": int(assignment_train_mask.sum().item()),
        "confidence_trainable_count": int(confidence_train_mask.sum().item()),
        "residual_bad_trainable_count": int(residual_bad.sum().item()),
        "residual_margin_trainable_count": int(margin_small.sum().item()),
        "residual_switch_count": int(residual_switch.sum().item()),
        "knn_trainable_count": int(knn_train_mask.sum().item()),
        "spatial_valid_count": diagnostic_spatial_valid_count,
        "final_trainable_count": int(train_mask.sum().item()),
        "assignment_residual_bad_quantile": float(assignment_residual_bad_quantile),
        "assignment_switch_improve_ratio": float(assignment_switch_improve_ratio),
        "part_residual_stats": part_stats,
        "knn": knn_stats,
        "pseudo_targets_used_for_training": False,
    }
    return train_mask, target_labels, spatial_target_labels, spatial_valid_mask, report


def fit_articulated_bases_from_stage1_gaussians(
    stage1: SceneModel,
    scene_path: str | Path,
    scene_norm_transform: torch.Tensor | None = None,
    scene_norm_scale: torch.Tensor | float | None = None,
    assignment_train_conf_thresh: float = 0.8,
    fit_core_min_conf: float = 0.7,
    fit_trim_quantile: float = 0.8,
    enable_assignment_refine: bool = True,
    assignment_refine_stride: int = 5,
    assignment_residual_bad_quantile: float = 0.8,
    assignment_switch_improve_ratio: float = 0.15,
    enable_assignment_knn_prior: bool = True,
    spatial_knn_k: int = 24,
    spatial_majority_ratio: float = 0.70,
    spatial_residual_slack_ratio: float = 0.10,
    assignment_refine_chunk_size: int = 2048,
    max_fit_points: int = 4096,
    min_fit_points: int = 128,
) -> tuple[
    ArticulatedMotionBasesArticulat3DStage2,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict,
]:
    device = stage1.fg.params["means"].device
    dtype = stage1.fg.params["means"].dtype
    logits = stage1.fg.params["motion_coefs"].detach()
    probs = F.softmax(logits, dim=-1)
    confidences, labels = probs.max(dim=-1)
    assignment_train_mask = confidences < assignment_train_conf_thresh

    num_bases = stage1.motion_bases.num_bases
    num_frames = stage1.motion_bases.num_frames
    candidates = load_joint_candidates_from_annotation(
        scene_path,
        device=device,
        dtype=dtype,
        scene_norm_transform=scene_norm_transform,
        scene_norm_scale=scene_norm_scale,
    )
    dynamic_parts = list(range(1, num_bases))
    if len(candidates) < len(dynamic_parts):
        guru.warning(
            f"joint_details.json provides {len(candidates)} dynamic candidates but "
            f"{len(dynamic_parts)} dynamic bases are needed; falling back to free-basis fitting"
        )
        bases, report = fit_articulated_bases_from_free_motion(
            stage1.motion_bases.params["rots"].detach(),
            stage1.motion_bases.params["transls"].detach(),
            joint_types=None,
        )
        (
            assignment_train_mask,
            assignment_target_labels,
            assignment_spatial_target_labels,
            assignment_spatial_valid_mask,
            assignment_refine_report,
        ) = refine_assignment_targets_from_stage1_residuals(
            stage1=stage1,
            bases=bases,
            labels=labels,
            confidences=confidences,
            assignment_train_mask=assignment_train_mask,
            assignment_train_conf_thresh=assignment_train_conf_thresh,
            enable_assignment_refine=enable_assignment_refine,
            assignment_refine_stride=assignment_refine_stride,
            assignment_residual_bad_quantile=assignment_residual_bad_quantile,
            assignment_switch_improve_ratio=assignment_switch_improve_ratio,
            enable_assignment_knn_prior=enable_assignment_knn_prior,
            spatial_knn_k=spatial_knn_k,
            spatial_majority_ratio=spatial_majority_ratio,
            spatial_residual_slack_ratio=spatial_residual_slack_ratio,
            trim_quantile=fit_trim_quantile,
            chunk_size=assignment_refine_chunk_size,
        )
        return bases, assignment_train_mask, assignment_target_labels, assignment_spatial_target_labels, assignment_spatial_valid_mask, {
            "mode": "free_basis_fallback",
            "rotation_fro_error": report.rotation_fro_error,
            "translation_l1_error": report.translation_l1_error,
            "joint_types": report.joint_types,
            "assignment_trainable_count": int(assignment_train_mask.sum().item()),
            "assignment_refine": assignment_refine_report,
        }

    ts = torch.arange(num_frames, device=device)
    part_points = {}
    part_anchors = {}
    part_lengths = {}
    for part_idx in range(num_bases):
        fit_indices = _select_fit_indices(
            labels,
            confidences,
            part_idx,
            min_conf=fit_core_min_conf,
            max_points=max_fit_points,
            min_points=min_fit_points,
        )
        if fit_indices.numel() == 0:
            raise ValueError(f"No gaussians assigned to stage2 part {part_idx}")
        with torch.no_grad():
            points, _ = stage1.compute_poses_fg(ts, inds=fit_indices)
            points = points.detach()
        part_points[part_idx] = points
        anchor, length = _part_viz_anchor_and_length(points)
        part_anchors[part_idx] = anchor
        part_lengths[part_idx] = length

    cost_matrix = []
    fit_cache = {}
    for part_idx in dynamic_parts:
        row = []
        points = part_points[part_idx]
        for cand_idx, candidate in enumerate(candidates):
            if candidate.joint_type_id == PRISMATIC_ID:
                values, pred = _fit_prismatic_points(points, candidate.axis, fit_trim_quantile)
            elif candidate.joint_type_id == REVOLUTE_ID:
                values, pred = _fit_revolute_points(
                    points, candidate.axis, candidate.pivot, fit_trim_quantile
                )
            else:
                continue
            stats = _trajectory_residual_stats(pred, points, fit_trim_quantile)
            fit_cache[(part_idx, cand_idx)] = (values, stats)
            row.append(stats["cost"])
        cost_matrix.append(row)

    best_perm = None
    best_cost = float("inf")
    for perm in itertools.permutations(range(len(candidates)), len(dynamic_parts)):
        cost = sum(cost_matrix[row_idx][cand_idx] for row_idx, cand_idx in enumerate(perm))
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
    if best_perm is None:
        raise RuntimeError("Unable to match stage2 parts to joint candidates")

    joint_type_ids = torch.full((num_bases,), STATIC_ID, dtype=torch.long, device=device)
    axis_raw = torch.zeros(num_bases, 3, dtype=dtype, device=device)
    pivot = torch.zeros(num_bases, 3, dtype=dtype, device=device)
    values = torch.zeros(num_bases, num_frames, dtype=dtype, device=device)
    viz_anchor = torch.zeros(num_bases, 3, dtype=dtype, device=device)
    viz_length = torch.ones(num_bases, dtype=dtype, device=device)
    axis_raw[0] = axis_raw.new_tensor([1.0, 0.0, 0.0])
    viz_anchor[0] = part_anchors[0]
    viz_length[0] = part_lengths[0]

    assignments = []
    for row_idx, part_idx in enumerate(dynamic_parts):
        cand_idx = best_perm[row_idx]
        candidate = candidates[cand_idx]
        fit_values, stats = fit_cache[(part_idx, cand_idx)]
        joint_type_ids[part_idx] = candidate.joint_type_id
        axis_raw[part_idx] = candidate.axis
        pivot[part_idx] = candidate.pivot if candidate.joint_type_id == REVOLUTE_ID else 0.0
        values[part_idx] = fit_values
        viz_anchor[part_idx] = (
            candidate.pivot if candidate.joint_type_id == REVOLUTE_ID else part_anchors[part_idx]
        )
        viz_length[part_idx] = part_lengths[part_idx]
        assignments.append(
            {
                "part_idx": part_idx,
                "candidate_index": candidate.source_index,
                "joint_type": candidate.joint_type,
                "cost": stats["cost"],
                "median": stats["median"],
                "mean": stats["mean"],
                "q90": stats["q90"],
                "q98": stats["q98"],
                "value_min": float(fit_values.min().item()),
                "value_max": float(fit_values.max().item()),
            }
        )

    bases = ArticulatedMotionBasesArticulat3DStage2(
        joint_type_ids=joint_type_ids,
        axis_raw=axis_raw,
        pivot=pivot,
        values=values,
        axis_prior=axis_raw.detach().clone(),
        pivot_prior=pivot.detach().clone(),
        init_axis=axis_raw.detach().clone(),
        init_pivot=pivot.detach().clone(),
        viz_anchor=viz_anchor,
        viz_length=viz_length,
    )
    report = {
        "mode": "joint_details_track_matching",
        "joint_types": bases.joint_types,
        "joint_candidate_path": str(Path(scene_path) / "joint_details.json"),
        "cost_matrix": cost_matrix,
        "assignments": assignments,
        "assignment_train_conf_thresh": assignment_train_conf_thresh,
        "assignment_trainable_count": int(assignment_train_mask.sum().item()),
        "num_gaussians": int(logits.shape[0]),
        "part_counts": torch.bincount(labels, minlength=num_bases).detach().cpu().tolist(),
    }
    (
        assignment_train_mask,
        assignment_target_labels,
        assignment_spatial_target_labels,
        assignment_spatial_valid_mask,
        assignment_refine_report,
    ) = refine_assignment_targets_from_stage1_residuals(
        stage1=stage1,
        bases=bases,
        labels=labels,
        confidences=confidences,
        assignment_train_mask=assignment_train_mask,
        assignment_train_conf_thresh=assignment_train_conf_thresh,
        enable_assignment_refine=enable_assignment_refine,
        assignment_refine_stride=assignment_refine_stride,
        assignment_residual_bad_quantile=assignment_residual_bad_quantile,
        assignment_switch_improve_ratio=assignment_switch_improve_ratio,
        enable_assignment_knn_prior=enable_assignment_knn_prior,
        spatial_knn_k=spatial_knn_k,
        spatial_majority_ratio=spatial_majority_ratio,
        spatial_residual_slack_ratio=spatial_residual_slack_ratio,
        trim_quantile=fit_trim_quantile,
        chunk_size=assignment_refine_chunk_size,
    )
    report["assignment_trainable_count"] = int(assignment_train_mask.sum().item())
    report["assignment_refine"] = assignment_refine_report
    return (
        bases,
        assignment_train_mask,
        assignment_target_labels,
        assignment_spatial_target_labels,
        assignment_spatial_valid_mask,
        report,
    )


def fit_articulated_bases_from_free_motion(
    rots_6d: torch.Tensor,
    transls: torch.Tensor,
    joint_types: list[str] | None = None,
) -> tuple[ArticulatedMotionBasesArticulat3DStage2, Stage2MotionReport]:
    R_free = cont_6d_to_rmat(rots_6d)
    K, T = transls.shape[:2]
    if joint_types is None:
        joint_types = [_infer_joint_type(R_free[k], transls[k]) for k in range(K)]
    if len(joint_types) != K:
        raise ValueError(f"joint_types length {len(joint_types)} does not match num bases {K}")

    joint_type_ids = []
    axis_raw = []
    pivot = []
    values = []
    for k, joint_type in enumerate(joint_types):
        joint_type = joint_type.lower()
        joint_id = JOINT_TYPE_TO_ID[joint_type]
        joint_type_ids.append(joint_id)
        if joint_id == REVOLUTE_ID:
            axis_k, pivot_k, values_k = _fit_revolute(R_free[k], transls[k])
        elif joint_id == PRISMATIC_ID:
            axis_k, values_k = _fit_prismatic(transls[k])
            pivot_k = transls.new_zeros(3)
        else:
            axis_k = transls.new_tensor([1.0, 0.0, 0.0])
            pivot_k = transls.new_zeros(3)
            values_k = transls.new_zeros(T)
        axis_raw.append(axis_k)
        pivot.append(pivot_k)
        values.append(values_k)

    bases = ArticulatedMotionBasesArticulat3DStage2(
        joint_type_ids=torch.tensor(joint_type_ids, device=transls.device),
        axis_raw=torch.stack(axis_raw, dim=0),
        pivot=torch.stack(pivot, dim=0),
        values=torch.stack(values, dim=0),
    )
    with torch.no_grad():
        R_fit, t_fit = bases.compute_base_transforms(torch.arange(T, device=transls.device))
        report = Stage2MotionReport(
            rotation_fro_error=(R_fit - R_free).norm(dim=(-1, -2)).mean().item(),
            translation_l1_error=(t_fit - transls).abs().mean().item(),
            joint_types=bases.joint_types,
        )
    return bases, report


def sharpen_motion_logits(logits: torch.Tensor, strength: float = 4.0) -> torch.Tensor:
    labels = logits.argmax(dim=-1)
    out = torch.full_like(logits, -strength)
    out.scatter_(1, labels[:, None], strength)
    return out


def initialize_stage2_model_from_stage1_checkpoint(
    stage1_ckpt_path: str | Path,
    scene_path: str | Path,
    device: torch.device,
    use_2dgs: bool = False,
    assignment_logit_strength: float = 4.0,
    scene_norm_transform: torch.Tensor | None = None,
    scene_norm_scale: torch.Tensor | float | None = None,
    assignment_train_conf_thresh: float = 0.8,
    fit_core_min_conf: float = 0.7,
    fit_trim_quantile: float = 0.8,
    enable_assignment_refine: bool = True,
    assignment_refine_stride: int = 5,
    assignment_residual_bad_quantile: float = 0.8,
    assignment_switch_improve_ratio: float = 0.15,
    enable_assignment_knn_prior: bool = True,
    spatial_knn_k: int = 24,
    spatial_majority_ratio: float = 0.70,
    spatial_residual_slack_ratio: float = 0.10,
    assignment_refine_chunk_size: int = 2048,
    assignment_graph_stride: int = 50,
    assignment_graph_motion_quantile: float = 0.8,
    gt_joint_axis_path: str | Path | None = None,
) -> tuple[SceneModelArticulat3DStage2, dict]:
    stage1_ckpt_path = Path(stage1_ckpt_path)
    if not stage1_ckpt_path.exists():
        raise FileNotFoundError(stage1_ckpt_path)
    ckpt = torch.load(stage1_ckpt_path, weights_only=False, map_location=device)
    stage1 = SceneModel.init_from_state_dict(ckpt["model"]).to(device)
    stage1.use_2dgs = use_2dgs

    (
        bases,
        assignment_train_mask,
        assignment_target_labels,
        assignment_spatial_target_labels,
        assignment_spatial_valid_mask,
        report_dict,
    ) = fit_articulated_bases_from_stage1_gaussians(
        stage1=stage1,
        scene_path=scene_path,
        scene_norm_transform=scene_norm_transform,
        scene_norm_scale=scene_norm_scale,
        assignment_train_conf_thresh=assignment_train_conf_thresh,
        fit_core_min_conf=fit_core_min_conf,
        fit_trim_quantile=fit_trim_quantile,
        enable_assignment_refine=enable_assignment_refine,
        assignment_refine_stride=assignment_refine_stride,
        assignment_residual_bad_quantile=assignment_residual_bad_quantile,
        assignment_switch_improve_ratio=assignment_switch_improve_ratio,
        enable_assignment_knn_prior=enable_assignment_knn_prior,
        spatial_knn_k=spatial_knn_k,
        spatial_majority_ratio=spatial_majority_ratio,
        spatial_residual_slack_ratio=spatial_residual_slack_ratio,
        assignment_refine_chunk_size=assignment_refine_chunk_size,
    )
    bases = bases.to(device)

    fg = stage1.fg
    if "motion_coefs" not in fg.params:
        raise KeyError("stage1 checkpoint does not contain fg.params.motion_coefs")
    with torch.no_grad():
        cano_ts = torch.zeros(1, dtype=torch.long, device=device)
        cano_means, cano_quats = stage1.compute_poses_fg(cano_ts)
        fg.params["means"] = nn.Parameter(cano_means[:, 0].detach().clone())
        fg.params["quats"] = nn.Parameter(cano_quats[:, 0].detach().clone())
        assignment_graph_edges, assignment_graph_weights, graph_report = (
            _build_assignment_graph_from_stage1(
                stage1=stage1,
                canonical_means=cano_means[:, 0].detach(),
                spatial_knn_k=spatial_knn_k,
                graph_frame_stride=assignment_graph_stride,
                graph_motion_quantile=assignment_graph_motion_quantile,
                chunk_size=assignment_refine_chunk_size,
            )
        )
        (
            gt_prismatic_segments,
            gt_revolute_segments,
            gt_revolute_pivots,
            gt_joint_viz_report,
        ) = load_gt_joint_viz_from_mobility(
            scene_path=scene_path,
            bases=bases,
            scene_norm_transform=scene_norm_transform,
            scene_norm_scale=scene_norm_scale,
            gt_joint_axis_path=gt_joint_axis_path,
        )
        bases.set_gt_joint_viz(
            gt_prismatic_segments=gt_prismatic_segments,
            gt_revolute_segments=gt_revolute_segments,
            gt_revolute_pivots=gt_revolute_pivots,
        )
    initial_logits = sharpen_motion_logits(
        fg.params["motion_coefs"].detach(),
        strength=assignment_logit_strength,
    )
    fg.params["motion_coefs"] = nn.Parameter(initial_logits.clone())
    initial_means = fg.params["means"].detach().clone()
    initial_opacities = fg.params["opacities"].detach().clone()
    opacity_train_mask = assignment_train_mask.detach().clone()
    means_train_mask = assignment_train_mask.detach().clone()

    model = SceneModelArticulat3DStage2(
        Ks=stage1.Ks.detach().clone(),
        w2cs=stage1.w2cs.detach().clone(),
        fg_params=fg,
        motion_bases=bases,
        camera_poses=None,
        bg_params=stage1.bg,
        use_2dgs=use_2dgs,
        initial_motion_logits=initial_logits,
        assignment_train_mask=assignment_train_mask,
        assignment_target_labels=assignment_target_labels,
        assignment_spatial_target_labels=assignment_spatial_target_labels,
        assignment_spatial_valid_mask=assignment_spatial_valid_mask,
        assignment_graph_edges=assignment_graph_edges,
        assignment_graph_weights=assignment_graph_weights,
        initial_fg_means=initial_means,
        initial_fg_opacities=initial_opacities,
        opacity_train_mask=opacity_train_mask,
        means_train_mask=means_train_mask,
    ).to(device)

    report_dict["stage1_ckpt_path"] = str(stage1_ckpt_path)
    report_dict["canonical_gaussians"] = "stage1_posed_frame_000000"
    assignment_diagnostics = report_dict.pop("assignment_refine", {})
    assignment_diagnostics["assignment_graph"] = graph_report
    assignment_diagnostics["pseudo_targets_used_for_training"] = False
    assignment_diagnostics["opacity_trainable_count"] = int(opacity_train_mask.sum().item())
    assignment_diagnostics["means_trainable_count"] = int(means_train_mask.sum().item())
    report_dict["assignment_diagnostics"] = assignment_diagnostics
    report_dict["gt_joint_viz"] = gt_joint_viz_report
    guru.info(
        "Initialized stage2 articulated bases: "
        f"joint_types={report_dict['joint_types']}, "
        f"mode={report_dict['mode']}, "
        f"assignment_trainable={report_dict['assignment_trainable_count']}"
    )
    return model, report_dict
