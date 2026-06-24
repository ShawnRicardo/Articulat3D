from __future__ import annotations

from dataclasses import dataclass

import roma
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from flow3d.params import GaussianParams, MotionBases, CameraPoses
from flow3d.scene_model import SceneModel


STATIC_ID = 0
REVOLUTE_ID = 1
PRISMATIC_ID = 2

JOINT_TYPE_TO_ID = {
    "static": STATIC_ID,
    "fixed": STATIC_ID,
    "base": STATIC_ID,
    "revolute": REVOLUTE_ID,
    "hinge": REVOLUTE_ID,
    "prismatic": PRISMATIC_ID,
    "slider": PRISMATIC_ID,
}
JOINT_ID_TO_TYPE = {
    STATIC_ID: "static",
    REVOLUTE_ID: "revolute",
    PRISMATIC_ID: "prismatic",
}


@dataclass
class Stage2MotionReport:
    rotation_fro_error: float
    translation_l1_error: float
    joint_types: list[str]


class ArticulatedMotionBasesArticulat3DStage2(nn.Module):
    """Joint-parameterized motion bases for stage2.

    The class keeps the same compute_transforms(ts, coefs) surface as
    flow3d.params.MotionBases so it can be used by SceneModel.render().
    Forward motion is physically generated from static/revolute/prismatic
    parameters and then selected by one-hot part assignment coefficients.
    """

    def __init__(
        self,
        joint_type_ids: Tensor,
        axis_raw: Tensor,
        pivot: Tensor,
        values: Tensor,
        axis_prior: Tensor | None = None,
        pivot_prior: Tensor | None = None,
        init_axis: Tensor | None = None,
        init_pivot: Tensor | None = None,
        viz_anchor: Tensor | None = None,
        viz_length: Tensor | None = None,
        gt_prismatic_segments: Tensor | None = None,
        gt_revolute_segments: Tensor | None = None,
        gt_revolute_pivots: Tensor | None = None,
    ):
        super().__init__()
        joint_type_ids = joint_type_ids.long()
        self.num_bases = int(joint_type_ids.shape[0])
        self.num_frames = int(values.shape[1])
        if axis_raw.shape != (self.num_bases, 3):
            raise ValueError(f"axis_raw must be {(self.num_bases, 3)}, got {tuple(axis_raw.shape)}")
        if pivot.shape != (self.num_bases, 3):
            raise ValueError(f"pivot must be {(self.num_bases, 3)}, got {tuple(pivot.shape)}")
        if values.shape != (self.num_bases, self.num_frames):
            raise ValueError(
                f"values must be {(self.num_bases, self.num_frames)}, got {tuple(values.shape)}"
            )
        for name, value, shape in [
            ("init_axis", init_axis, (self.num_bases, 3)),
            ("init_pivot", init_pivot, (self.num_bases, 3)),
            ("viz_anchor", viz_anchor, (self.num_bases, 3)),
            ("viz_length", viz_length, (self.num_bases,)),
        ]:
            if value is not None and value.shape != shape:
                raise ValueError(f"{name} must be {shape}, got {tuple(value.shape)}")

        self.register_buffer("joint_type_ids", joint_type_ids)
        self.params = nn.ParameterDict(
            {
                "axis_raw": nn.Parameter(axis_raw),
                "pivot": nn.Parameter(pivot),
                "joint_values": nn.Parameter(values),
            }
        )
        if axis_prior is None:
            axis_prior = axis_raw.detach().clone()
        if pivot_prior is None:
            pivot_prior = pivot.detach().clone()
        self.register_buffer("axis_prior", axis_prior)
        self.register_buffer("pivot_prior", pivot_prior)
        if init_axis is None:
            init_axis = axis_raw.detach().clone()
        if init_pivot is None:
            init_pivot = pivot.detach().clone()
        if viz_anchor is None:
            viz_anchor = pivot.detach().clone()
        if viz_length is None:
            viz_length = torch.ones(self.num_bases, dtype=axis_raw.dtype, device=axis_raw.device)
        self.register_buffer("init_axis", F.normalize(init_axis, dim=-1, eps=1e-8))
        self.register_buffer("init_pivot", init_pivot)
        self.register_buffer("viz_anchor", viz_anchor)
        self.register_buffer("viz_length", viz_length.clamp(min=1e-6))
        self.set_gt_joint_viz(
            gt_prismatic_segments=gt_prismatic_segments,
            gt_revolute_segments=gt_revolute_segments,
            gt_revolute_pivots=gt_revolute_pivots,
        )

    @property
    def joint_types(self) -> list[str]:
        return [JOINT_ID_TO_TYPE[int(i)] for i in self.joint_type_ids.detach().cpu().tolist()]

    @staticmethod
    def init_from_state_dict(state_dict: dict[str, Tensor], prefix: str = "motion_bases."):
        req = [
            f"{prefix}joint_type_ids",
            f"{prefix}params.axis_raw",
            f"{prefix}params.pivot",
            f"{prefix}params.joint_values",
        ]
        missing = [k for k in req if k not in state_dict]
        if missing:
            raise KeyError(f"Missing stage2 motion basis keys: {missing}")
        return ArticulatedMotionBasesArticulat3DStage2(
            joint_type_ids=state_dict[f"{prefix}joint_type_ids"],
            axis_raw=state_dict[f"{prefix}params.axis_raw"],
            pivot=state_dict[f"{prefix}params.pivot"],
            values=state_dict[f"{prefix}params.joint_values"],
            axis_prior=state_dict.get(f"{prefix}axis_prior"),
            pivot_prior=state_dict.get(f"{prefix}pivot_prior"),
            init_axis=state_dict.get(f"{prefix}init_axis"),
            init_pivot=state_dict.get(f"{prefix}init_pivot"),
            viz_anchor=state_dict.get(f"{prefix}viz_anchor"),
            viz_length=state_dict.get(f"{prefix}viz_length"),
            gt_prismatic_segments=state_dict.get(f"{prefix}gt_prismatic_segments"),
            gt_revolute_segments=state_dict.get(f"{prefix}gt_revolute_segments"),
            gt_revolute_pivots=state_dict.get(f"{prefix}gt_revolute_pivots"),
        )

    def set_gt_joint_viz(
        self,
        gt_prismatic_segments: Tensor | None = None,
        gt_revolute_segments: Tensor | None = None,
        gt_revolute_pivots: Tensor | None = None,
    ):
        device = self.params["axis_raw"].device
        dtype = self.params["axis_raw"].dtype
        if gt_prismatic_segments is None:
            gt_prismatic_segments = torch.empty(0, 2, 3, device=device, dtype=dtype)
        if gt_revolute_segments is None:
            gt_revolute_segments = torch.empty(0, 2, 3, device=device, dtype=dtype)
        if gt_revolute_pivots is None:
            gt_revolute_pivots = torch.empty(0, 3, device=device, dtype=dtype)
        if gt_prismatic_segments.ndim != 3 or gt_prismatic_segments.shape[1:] != (2, 3):
            raise ValueError(
                "gt_prismatic_segments must have shape (N, 2, 3), got "
                f"{tuple(gt_prismatic_segments.shape)}"
            )
        if gt_revolute_segments.ndim != 3 or gt_revolute_segments.shape[1:] != (2, 3):
            raise ValueError(
                "gt_revolute_segments must have shape (N, 2, 3), got "
                f"{tuple(gt_revolute_segments.shape)}"
            )
        if gt_revolute_pivots.ndim != 2 or gt_revolute_pivots.shape[-1] != 3:
            raise ValueError(
                "gt_revolute_pivots must have shape (N, 3), got "
                f"{tuple(gt_revolute_pivots.shape)}"
            )
        def _set_buffer(name: str, value: Tensor):
            value = value.to(device=device, dtype=dtype)
            if name in self._buffers:
                setattr(self, name, value)
            else:
                self.register_buffer(name, value)

        _set_buffer("gt_prismatic_segments", gt_prismatic_segments)
        _set_buffer("gt_revolute_segments", gt_revolute_segments)
        _set_buffer("gt_revolute_pivots", gt_revolute_pivots)

    def normalized_axes(self) -> Tensor:
        return F.normalize(self.params["axis_raw"], dim=-1, eps=1e-8)

    def compute_base_transforms(self, ts: Tensor) -> tuple[Tensor, Tensor]:
        device = self.params["joint_values"].device
        dtype = self.params["joint_values"].dtype
        ts = ts.long()
        B = int(ts.shape[0])
        eye = torch.eye(3, device=device, dtype=dtype)
        axes = self.normalized_axes()
        values = self.params["joint_values"][:, ts]
        pivots = self.params["pivot"]

        rots = []
        transls = []
        for k in range(self.num_bases):
            joint_type = int(self.joint_type_ids[k].item())
            if joint_type == REVOLUTE_ID:
                rot = roma.rotvec_to_rotmat(values[k, :, None] * axes[k][None])
                transl = pivots[k][None] - torch.einsum("bij,j->bi", rot, pivots[k])
            elif joint_type == PRISMATIC_ID:
                rot = eye[None].expand(B, 3, 3)
                transl = values[k, :, None] * axes[k][None]
            else:
                rot = eye[None].expand(B, 3, 3)
                transl = torch.zeros(B, 3, device=device, dtype=dtype)
            rots.append(rot)
            transls.append(transl)
        return torch.stack(rots, dim=0), torch.stack(transls, dim=0)

    def compute_transforms(self, ts: Tensor, coefs: Tensor) -> Tensor:
        base_rots, base_transls = self.compute_base_transforms(ts)
        rots = torch.einsum("gk,kbij->gbij", coefs, base_rots)
        transls = torch.einsum("gk,kbi->gbi", coefs, base_transls)
        return torch.cat([rots, transls[..., None]], dim=-1)

    def value_smoothness_loss(self, segment_length: int = 0) -> Tensor:
        values = self.params["joint_values"]
        if values.shape[1] < 3:
            return values.new_zeros(())

        losses = []
        if segment_length <= 0 or values.shape[1] <= segment_length:
            accel = 2 * values[:, 1:-1] - values[:, :-2] - values[:, 2:]
            losses.append(accel.abs().mean())
        else:
            for start in range(0, values.shape[1], segment_length):
                end = min(start + segment_length, values.shape[1])
                if end - start >= 3:
                    seg = values[:, start:end]
                    accel = 2 * seg[:, 1:-1] - seg[:, :-2] - seg[:, 2:]
                    losses.append(accel.abs().mean())
        if not losses:
            return values.new_zeros(())
        return torch.stack(losses).mean()

    def axis_prior_loss(self) -> Tensor:
        movable = self.joint_type_ids != STATIC_ID
        if not movable.any():
            return self.params["axis_raw"].new_zeros(())
        axes = self.normalized_axes()[movable]
        prior = F.normalize(self.axis_prior[movable], dim=-1, eps=1e-8)
        return (1.0 - (axes * prior).sum(dim=-1).abs()).mean()

    def pivot_prior_loss(self) -> Tensor:
        revolute = self.joint_type_ids == REVOLUTE_ID
        if not revolute.any():
            return self.params["pivot"].new_zeros(())
        return F.mse_loss(self.params["pivot"][revolute], self.pivot_prior[revolute])

    def revolute_pivot_residual(self, ts: Tensor | None = None) -> Tensor:
        revolute = self.joint_type_ids == REVOLUTE_ID
        if not revolute.any():
            return self.params["pivot"].new_zeros(())
        if ts is None:
            ts = torch.arange(self.num_frames, device=self.params["joint_values"].device)
        rots, transls = self.compute_base_transforms(ts)
        pivots = self.params["pivot"][:, None]
        moved = torch.einsum("kbij,kj->kbi", rots, self.params["pivot"]) + transls
        return (moved[revolute] - pivots[revolute]).norm(dim=-1).mean()

    def prismatic_axis_residual(self, ts: Tensor | None = None) -> Tensor:
        prismatic = self.joint_type_ids == PRISMATIC_ID
        if not prismatic.any():
            return self.params["axis_raw"].new_zeros(())
        if ts is None:
            ts = torch.arange(self.num_frames, device=self.params["joint_values"].device)
        _, transls = self.compute_base_transforms(ts)
        axes = self.normalized_axes()
        parallel = (transls * axes[:, None]).sum(dim=-1, keepdim=True) * axes[:, None]
        return (transls[prismatic] - parallel[prismatic]).norm(dim=-1).mean()

    @torch.no_grad()
    def get_joint_viz_segments(self, use_initial: bool = False) -> dict[str, Tensor]:
        if use_initial:
            axes = F.normalize(self.init_axis, dim=-1, eps=1e-8)
            pivots = self.init_pivot
        else:
            axes = self.normalized_axes()
            pivots = self.params["pivot"]

        anchors = self.viz_anchor.clone()
        revolute = self.joint_type_ids == REVOLUTE_ID
        anchors[revolute] = pivots[revolute]
        half_lengths = 0.5 * self.viz_length[:, None]
        segments = torch.stack(
            (anchors - axes * half_lengths, anchors + axes * half_lengths), dim=1
        )
        prismatic = self.joint_type_ids == PRISMATIC_ID
        return {
            "prismatic_segments": segments[prismatic].detach(),
            "revolute_segments": segments[revolute].detach(),
            "revolute_pivots": pivots[revolute].detach(),
        }

    @torch.no_grad()
    def get_gt_joint_viz_segments(self) -> dict[str, Tensor]:
        return {
            "prismatic_segments": self.gt_prismatic_segments.detach(),
            "revolute_segments": self.gt_revolute_segments.detach(),
            "revolute_pivots": self.gt_revolute_pivots.detach(),
        }


class SceneModelArticulat3DStage2(SceneModel):
    def __init__(
        self,
        Ks: Tensor,
        w2cs: Tensor,
        fg_params: GaussianParams,
        motion_bases: ArticulatedMotionBasesArticulat3DStage2,
        camera_poses: CameraPoses | None = None,
        bg_params: GaussianParams | None = None,
        use_2dgs: bool = False,
        assignment_tau: float = 1.0,
        initial_motion_logits: Tensor | None = None,
        assignment_train_mask: Tensor | None = None,
        assignment_target_labels: Tensor | None = None,
        assignment_spatial_target_labels: Tensor | None = None,
        assignment_spatial_valid_mask: Tensor | None = None,
        assignment_graph_edges: Tensor | None = None,
        assignment_graph_weights: Tensor | None = None,
        initial_fg_means: Tensor | None = None,
        initial_fg_opacities: Tensor | None = None,
        opacity_train_mask: Tensor | None = None,
        means_train_mask: Tensor | None = None,
        ghost_visible_count: Tensor | None = None,
        ghost_outside_count: Tensor | None = None,
        ghost_depth_bad_count: Tensor | None = None,
        ghost_score: Tensor | None = None,
        reassign_attempt_count: Tensor | None = None,
        reassign_last_step: Tensor | None = None,
        reassign_last_from_label: Tensor | None = None,
        reassign_last_to_label: Tensor | None = None,
        part_prior_centers: Tensor | None = None,
        part_prior_axes: Tensor | None = None,
        part_prior_scales: Tensor | None = None,
        part_prior_valid_mask: Tensor | None = None,
    ):
        super().__init__(Ks, w2cs, fg_params, motion_bases, camera_poses, bg_params, use_2dgs)
        self.assignment_tau = assignment_tau
        if initial_motion_logits is None:
            initial_motion_logits = fg_params.params["motion_coefs"].detach().clone()
        self.register_buffer("initial_motion_logits", initial_motion_logits)
        num_gaussians = fg_params.params["motion_coefs"].shape[0]
        if assignment_target_labels is None:
            assignment_target_labels = initial_motion_logits.argmax(dim=-1)
        if assignment_spatial_target_labels is None:
            assignment_spatial_target_labels = assignment_target_labels.detach().clone()
        if assignment_spatial_valid_mask is None:
            assignment_spatial_valid_mask = torch.zeros(
                num_gaussians,
                dtype=torch.bool,
                device=fg_params.params["motion_coefs"].device,
            )
        if assignment_train_mask is None:
            assignment_train_mask = torch.ones(
                num_gaussians,
                dtype=torch.bool,
                device=fg_params.params["motion_coefs"].device,
            )
        self.register_buffer("assignment_target_labels", assignment_target_labels.long())
        self.register_buffer(
            "assignment_spatial_target_labels", assignment_spatial_target_labels.long()
        )
        self.register_buffer(
            "assignment_spatial_valid_mask", assignment_spatial_valid_mask.bool()
        )
        if assignment_graph_edges is None:
            assignment_graph_edges = torch.empty(
                0,
                2,
                dtype=torch.long,
                device=fg_params.params["motion_coefs"].device,
            )
        if assignment_graph_weights is None:
            assignment_graph_weights = torch.empty(
                0,
                dtype=fg_params.params["motion_coefs"].dtype,
                device=fg_params.params["motion_coefs"].device,
            )
        if initial_fg_means is None:
            initial_fg_means = fg_params.params["means"].detach().clone()
        if initial_fg_opacities is None:
            initial_fg_opacities = fg_params.params["opacities"].detach().clone()
        if opacity_train_mask is None:
            opacity_train_mask = assignment_train_mask.detach().clone()
        if means_train_mask is None:
            means_train_mask = assignment_train_mask.detach().clone()
        if ghost_visible_count is None:
            ghost_visible_count = torch.zeros(
                num_gaussians,
                dtype=fg_params.params["means"].dtype,
                device=fg_params.params["means"].device,
            )
        if ghost_outside_count is None:
            ghost_outside_count = torch.zeros_like(ghost_visible_count)
        if ghost_depth_bad_count is None:
            ghost_depth_bad_count = torch.zeros_like(ghost_visible_count)
        if ghost_score is None:
            ghost_score = torch.zeros_like(ghost_visible_count)
        if reassign_attempt_count is None:
            reassign_attempt_count = torch.zeros(
                num_gaussians,
                dtype=torch.long,
                device=fg_params.params["means"].device,
            )
        if reassign_last_step is None:
            reassign_last_step = torch.full(
                (num_gaussians,),
                -1,
                dtype=torch.long,
                device=fg_params.params["means"].device,
            )
        if reassign_last_from_label is None:
            reassign_last_from_label = torch.full_like(reassign_last_step, -1)
        if reassign_last_to_label is None:
            reassign_last_to_label = torch.full_like(reassign_last_step, -1)
        num_bases = int(fg_params.params["motion_coefs"].shape[-1])
        dtype = fg_params.params["means"].dtype
        device = fg_params.params["means"].device
        if part_prior_centers is None:
            part_prior_centers = torch.zeros(num_bases, 3, dtype=dtype, device=device)
        if part_prior_axes is None:
            part_prior_axes = torch.eye(3, dtype=dtype, device=device)[None].repeat(
                num_bases, 1, 1
            )
        if part_prior_scales is None:
            part_prior_scales = torch.ones(num_bases, 3, dtype=dtype, device=device)
        if part_prior_valid_mask is None:
            part_prior_valid_mask = torch.zeros(num_bases, dtype=torch.bool, device=device)
        self.register_buffer("assignment_graph_edges", assignment_graph_edges.long())
        self.register_buffer(
            "assignment_graph_weights",
            assignment_graph_weights.to(dtype=fg_params.params["motion_coefs"].dtype),
        )
        self.register_buffer("initial_fg_means", initial_fg_means)
        self.register_buffer("initial_fg_opacities", initial_fg_opacities)
        self.register_buffer("opacity_train_mask", opacity_train_mask.bool())
        self.register_buffer("means_train_mask", means_train_mask.bool())
        self.register_buffer("assignment_train_mask", assignment_train_mask.bool())
        self.register_buffer("ghost_visible_count", ghost_visible_count)
        self.register_buffer("ghost_outside_count", ghost_outside_count)
        self.register_buffer("ghost_depth_bad_count", ghost_depth_bad_count)
        self.register_buffer("ghost_score", ghost_score)
        self.register_buffer("reassign_attempt_count", reassign_attempt_count.long())
        self.register_buffer("reassign_last_step", reassign_last_step.long())
        self.register_buffer("reassign_last_from_label", reassign_last_from_label.long())
        self.register_buffer("reassign_last_to_label", reassign_last_to_label.long())
        self.register_buffer("part_prior_centers", part_prior_centers)
        self.register_buffer("part_prior_axes", part_prior_axes)
        self.register_buffer("part_prior_scales", part_prior_scales.clamp_min(1e-6))
        self.register_buffer("part_prior_valid_mask", part_prior_valid_mask.bool())
        self._assignment_grad_hook = None
        self._means_grad_hook = None
        self._opacity_grad_hook = None
        self.assignment_grad_mask_active = True
        self.means_grad_enabled = False
        self.opacity_grad_enabled = False
        self.stochastic_assignment = False
        self._register_assignment_grad_mask()
        self._register_gaussian_attr_grad_masks()

    def set_assignment_temperature(self, tau: float):
        self.assignment_tau = float(tau)

    def set_stochastic_assignment(self, enabled: bool):
        self.stochastic_assignment = bool(enabled)

    def _register_assignment_grad_mask(self):
        if "motion_coefs" not in self.fg.params:
            return
        if self._assignment_grad_hook is not None:
            self._assignment_grad_hook.remove()
        param = self.fg.params["motion_coefs"]

        def _mask_grad(grad: Tensor) -> Tensor:
            if not self.assignment_grad_mask_active:
                return grad
            mask = self.assignment_train_mask.to(device=grad.device, dtype=grad.dtype)
            return grad * mask[:, None]

        self._assignment_grad_hook = param.register_hook(_mask_grad)

    def _register_gaussian_attr_grad_masks(self):
        if "means" in self.fg.params:
            if self._means_grad_hook is not None:
                self._means_grad_hook.remove()
            means_param = self.fg.params["means"]

            def _mask_means_grad(grad: Tensor) -> Tensor:
                if not self.means_grad_enabled:
                    return torch.zeros_like(grad)
                mask = self.means_train_mask.to(device=grad.device, dtype=grad.dtype)
                return grad * mask[:, None]

            self._means_grad_hook = means_param.register_hook(_mask_means_grad)
        if "opacities" in self.fg.params:
            if self._opacity_grad_hook is not None:
                self._opacity_grad_hook.remove()
            opacity_param = self.fg.params["opacities"]

            def _mask_opacity_grad(grad: Tensor) -> Tensor:
                if not self.opacity_grad_enabled:
                    return torch.zeros_like(grad)
                mask = self.opacity_train_mask.to(device=grad.device, dtype=grad.dtype)
                while mask.ndim < grad.ndim:
                    mask = mask[..., None]
                return grad * mask

            self._opacity_grad_hook = opacity_param.register_hook(_mask_opacity_grad)

    def set_assignment_grad_mask_active(self, active: bool):
        self.assignment_grad_mask_active = bool(active)

    def set_gaussian_attr_grad_enabled(self, means: bool = False, opacities: bool = False):
        self.means_grad_enabled = bool(means)
        self.opacity_grad_enabled = bool(opacities)

    def set_assignment_train_mask(self, mask: Tensor):
        if mask.shape != self.assignment_train_mask.shape:
            raise ValueError(
                f"assignment train mask shape {tuple(mask.shape)} does not match "
                f"{tuple(self.assignment_train_mask.shape)}"
            )
        self.assignment_train_mask.copy_(mask.to(self.assignment_train_mask.device).bool())

    def get_motion_coefs(self, inds: Tensor | None = None, deterministic: bool | None = None) -> Tensor:
        logits = self.fg.params["motion_coefs"]
        if inds is not None:
            logits = logits[inds]
        if deterministic is None:
            deterministic = (not self.training) or (not torch.is_grad_enabled())
        if deterministic or not self.stochastic_assignment:
            labels = logits.argmax(dim=-1)
            return F.one_hot(labels, num_classes=logits.shape[-1]).to(dtype=logits.dtype)
        return F.gumbel_softmax(logits, tau=self.assignment_tau, hard=True, dim=-1)

    def assignment_prior_loss(self) -> Tensor:
        logits = self.fg.params["motion_coefs"]
        if self.assignment_target_labels.shape[0] != logits.shape[0]:
            return logits.new_zeros(())
        return F.cross_entropy(logits, self.assignment_target_labels.to(logits.device))

    def assignment_spatial_prior_loss(self) -> Tensor:
        logits = self.fg.params["motion_coefs"]
        if self.assignment_spatial_target_labels.shape[0] != logits.shape[0]:
            return logits.new_zeros(())
        mask = (self.assignment_train_mask & self.assignment_spatial_valid_mask).clone()
        if not mask.any():
            return logits.new_zeros(())
        labels = self.assignment_spatial_target_labels.to(logits.device)
        return F.cross_entropy(logits[mask], labels[mask])

    def assignment_graph_smooth_loss(self) -> Tensor:
        logits = self.fg.params["motion_coefs"]
        edges = self.assignment_graph_edges
        if edges.numel() == 0:
            return logits.new_zeros(())
        probs = F.softmax(logits, dim=-1)
        src = edges[:, 0].to(logits.device)
        dst = edges[:, 1].to(logits.device)
        weights = self.assignment_graph_weights.to(device=logits.device, dtype=logits.dtype)
        if weights.numel() != src.numel():
            weights = torch.ones(src.shape[0], device=logits.device, dtype=logits.dtype)
        diff = (probs[src] - probs[dst]).pow(2).sum(dim=-1)
        return (diff * weights).sum() / weights.sum().clamp_min(1e-8)

    def assignment_entropy_loss(self) -> Tensor:
        logits = self.fg.params["motion_coefs"]
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(probs * log_probs).sum(dim=-1).mean()

    def opacity_anchor_loss(self) -> Tensor:
        opacities = self.fg.params["opacities"]
        if self.initial_fg_opacities.shape != opacities.shape:
            return opacities.new_zeros(())
        mask = self.opacity_train_mask.to(opacities.device).clone()
        if not mask.any():
            return opacities.new_zeros(())
        return F.mse_loss(opacities[mask], self.initial_fg_opacities.to(opacities.device)[mask])

    def means_anchor_loss(self) -> Tensor:
        means = self.fg.params["means"]
        if self.initial_fg_means.shape != means.shape:
            return means.new_zeros(())
        mask = self.means_train_mask.to(means.device).clone()
        if not mask.any():
            return means.new_zeros(())
        return F.mse_loss(means[mask], self.initial_fg_means.to(means.device)[mask])

    @torch.no_grad()
    def update_ghost_scores(
        self,
        means_fg: Tensor,
        w2cs: Tensor,
        Ks: Tensor,
        masks: Tensor,
        dilated_masks: Tensor,
        depths: Tensor,
        depth_abs_thresh: float,
        depth_rel_thresh: float,
        score_thresh_for_opacity: float,
    ):
        if means_fg.numel() == 0:
            return
        B, N, _ = means_fg.shape
        H, W = masks.shape[-2:]
        device = means_fg.device
        dtype = means_fg.dtype
        visible = torch.zeros(N, device=device, dtype=dtype)
        outside = torch.zeros(N, device=device, dtype=dtype)
        depth_bad = torch.zeros(N, device=device, dtype=dtype)
        ones = torch.ones(N, 1, device=device, dtype=dtype)

        for b in range(B):
            pts_h = torch.cat([means_fg[b], ones], dim=-1)
            cam = torch.einsum("ij,nj->ni", w2cs[b, :3], pts_h)
            z = cam[:, 2]
            valid_z = z > 1e-6
            x = Ks[b, 0, 0] * (cam[:, 0] / z.clamp_min(1e-6)) + Ks[b, 0, 2]
            y = Ks[b, 1, 1] * (cam[:, 1] / z.clamp_min(1e-6)) + Ks[b, 1, 2]
            xi = torch.round(x).long()
            yi = torch.round(y).long()
            in_img = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H) & valid_z
            if not in_img.any():
                continue

            idx = torch.where(in_img)[0]
            px = xi[idx]
            py = yi[idx]
            visible[idx] += 1.0
            fg_at = masks[b, py, px] > 0.5
            dilated_at = dilated_masks[b, py, px] > 0.5
            outside[idx] += (~dilated_at).to(dtype)

            gt_depth = depths[b, py, px].to(dtype)
            z_sel = z[idx]
            depth_abs = (z_sel - gt_depth).abs()
            depth_rel = depth_abs / gt_depth.abs().clamp_min(1e-4)
            depth_bad_at = fg_at & (
                (depth_abs > float(depth_abs_thresh))
                & (depth_rel > float(depth_rel_thresh))
            )
            depth_bad[idx] += depth_bad_at.to(dtype)

        self.ghost_visible_count += visible
        self.ghost_outside_count += outside
        self.ghost_depth_bad_count += depth_bad
        self.ghost_score.copy_(
            (self.ghost_outside_count + self.ghost_depth_bad_count)
            / self.ghost_visible_count.clamp_min(1.0)
        )
        ghost_opacity_mask = self.ghost_score >= float(score_thresh_for_opacity)
        if ghost_opacity_mask.any():
            self.opacity_train_mask |= ghost_opacity_mask.to(self.opacity_train_mask.device)

    def ghost_opacity_decay_loss(
        self,
        score_thresh: float,
        min_visible: int,
    ) -> Tensor:
        opacities = self.fg.get_opacities()
        mask = (
            (self.ghost_score.to(opacities.device) >= float(score_thresh))
            & (self.ghost_visible_count.to(opacities.device) >= float(min_visible))
        ).clone()
        if not mask.any():
            return opacities.new_zeros(())
        score = self.ghost_score.to(device=opacities.device, dtype=opacities.dtype)
        vals = opacities.reshape(opacities.shape[0], -1).mean(dim=-1)
        return (vals[mask] * score[mask]).mean()

    @torch.no_grad()
    def update_part_priors(
        self,
        conf_thresh: float,
        opacity_thresh: float,
        ghost_score_thresh: float,
        min_points: int,
        scale_min: float,
        center_mode: str = "robust_obb",
        obb_low_quantile: float = 0.05,
        obb_high_quantile: float = 0.95,
    ) -> dict:
        logits = self.fg.params["motion_coefs"].detach()
        means = self.fg.params["means"].detach()
        probs = F.softmax(logits, dim=-1)
        conf, labels = probs.max(dim=-1)
        opacity = self.fg.get_opacities().detach().reshape(self.num_fg_gaussians, -1).mean(dim=-1)
        ghost = self.ghost_score.detach().to(means.device)

        centers = torch.zeros_like(self.part_prior_centers)
        axes_out = torch.eye(3, device=means.device, dtype=means.dtype)[None].repeat(
            self.num_motion_bases, 1, 1
        )
        scales = torch.ones_like(self.part_prior_scales)
        valid = torch.zeros_like(self.part_prior_valid_mask)
        counts: list[int] = []
        part_counts: list[int] = []
        core_counts: list[int] = []
        median_conf_thresholds: list[float] = []
        fallback_modes: list[str] = []
        obb_q05: list[list[float]] = []
        obb_q95: list[list[float]] = []
        obb_center_shift_norm: list[float] = []
        obb_low = float(obb_low_quantile)
        obb_high = float(obb_high_quantile)
        if not (0.0 <= obb_low < obb_high <= 1.0):
            raise ValueError(
                "OBB quantiles must satisfy 0 <= low < high <= 1, got "
                f"{obb_low_quantile}, {obb_high_quantile}"
            )

        for k in range(self.num_motion_bases):
            part_mask = labels == k
            part_count = int(part_mask.sum().item())
            part_counts.append(part_count)
            if part_count == 0:
                counts.append(0)
                core_counts.append(0)
                median_conf_thresholds.append(float("nan"))
                fallback_modes.append("empty_part")
                obb_q05.append([float("nan")] * 3)
                obb_q95.append([float("nan")] * 3)
                obb_center_shift_norm.append(float("nan"))
                continue
            part_conf = conf[part_mask]
            conf_cutoff = part_conf.median()
            median_conf_thresholds.append(float(conf_cutoff.item()))

            core_mask = (
                part_mask
                & (conf >= conf_cutoff)
                & (opacity >= float(opacity_thresh))
                & (ghost < float(ghost_score_thresh))
            )
            fallback_mode = "median_conf_opacity_low_ghost"
            if int(core_mask.sum().item()) < int(min_points):
                core_mask = (
                    part_mask
                    & (conf >= conf_cutoff)
                    & (opacity >= float(opacity_thresh))
                )
                fallback_mode = "median_conf_opacity"
            if int(core_mask.sum().item()) < int(min_points):
                part_indices = torch.nonzero(part_mask, as_tuple=False).flatten()
                order = torch.argsort(conf[part_indices], descending=True)
                take = min(max(int(min_points), 3), int(part_indices.numel()))
                core_mask = torch.zeros_like(part_mask)
                core_mask[part_indices[order[:take]]] = True
                fallback_mode = "top_confidence_min_points"
            if int(core_mask.sum().item()) < 3:
                core_mask = part_mask
                fallback_mode = "all_part_points"

            pts = means[core_mask]
            counts.append(int(pts.shape[0]))
            core_counts.append(int(core_mask.sum().item()))
            fallback_modes.append(fallback_mode)
            if pts.shape[0] < 3:
                obb_q05.append([float("nan")] * 3)
                obb_q95.append([float("nan")] * 3)
                obb_center_shift_norm.append(float("nan"))
                continue

            median_center = pts.median(dim=0).values
            centered = pts - median_center
            cov = centered.T @ centered / float(max(pts.shape[0] - 1, 1))
            try:
                eigvals, eigvecs = torch.linalg.eigh(cov)
                order = torch.argsort(eigvals, descending=True)
                axes = eigvecs[:, order]
            except RuntimeError:
                axes = torch.eye(3, device=means.device, dtype=means.dtype)
            local = centered @ axes
            if center_mode == "median":
                center = median_center
                low = -torch.quantile(local.abs(), 0.90, dim=0)
                high = torch.quantile(local.abs(), 0.90, dim=0)
            else:
                low = torch.quantile(local, obb_low, dim=0)
                high = torch.quantile(local, obb_high, dim=0)
                center_local = 0.5 * (low + high)
                center = median_center + center_local @ axes.T
            extent = (0.5 * (high - low)).clamp_min(float(scale_min))

            centers[k] = center
            axes_out[k] = axes
            scales[k] = extent
            valid[k] = True
            obb_q05.append(low.detach().cpu().tolist())
            obb_q95.append(high.detach().cpu().tolist())
            obb_center_shift_norm.append(float((center - median_center).norm().item()))

        self.part_prior_centers.copy_(centers)
        self.part_prior_axes.copy_(axes_out)
        self.part_prior_scales.copy_(scales.clamp_min(float(scale_min)))
        self.part_prior_valid_mask.copy_(valid)
        return {
            "valid_count": int(valid.sum().item()),
            "center_mode": "robust_obb_q05_q95" if center_mode != "median" else "median",
            "obb_low_quantile": obb_low,
            "obb_high_quantile": obb_high,
            "confidence_threshold_mode": "per_part_median",
            "legacy_conf_thresh_ignored": float(conf_thresh),
            "part_counts": part_counts,
            "counts": counts,
            "core_counts": core_counts,
            "median_conf_thresholds": median_conf_thresholds,
            "fallback_modes": fallback_modes,
            "obb_q05": obb_q05,
            "obb_q95": obb_q95,
            "obb_center_shift_norm": obb_center_shift_norm,
            "centers": centers.detach().cpu().tolist(),
            "scales": scales.detach().cpu().tolist(),
        }

    @torch.no_grad()
    def get_part_prior_centers(
        self,
        ts: int | Tensor | None = None,
        posed: bool = True,
    ) -> tuple[Tensor, Tensor]:
        valid = self.part_prior_valid_mask.detach()
        centers = self.part_prior_centers.detach()
        if not posed:
            return centers[valid], valid
        if ts is None:
            ts_tensor = torch.zeros(
                1,
                device=centers.device,
                dtype=torch.long,
            )
        elif isinstance(ts, Tensor):
            ts_tensor = ts.to(device=centers.device).long().reshape(1)
        else:
            ts_tensor = torch.tensor([int(ts)], device=centers.device, dtype=torch.long)
        ts_tensor = ts_tensor.clamp(min=0, max=self.motion_bases.num_frames - 1)
        base_rots, base_transls = self.motion_bases.compute_base_transforms(ts_tensor)
        posed_centers = (
            torch.einsum("kij,kj->ki", base_rots[:, 0], centers)
            + base_transls[:, 0]
        )
        return posed_centers[valid], valid

    @torch.no_grad()
    def anisotropic_part_prior_cost(self, points: Tensor) -> Tensor:
        if points.numel() == 0:
            return points.new_zeros((0, self.num_motion_bases))
        centers = self.part_prior_centers.to(device=points.device, dtype=points.dtype)
        axes = self.part_prior_axes.to(device=points.device, dtype=points.dtype)
        scales = self.part_prior_scales.to(device=points.device, dtype=points.dtype).clamp_min(1e-6)
        delta = points[:, None, :] - centers[None]
        local = torch.einsum("nkd,kde->nke", delta, axes)
        normalized = local.abs() / scales[None]
        outside = torch.clamp(normalized - 1.0, min=0.0).pow(2).sum(dim=-1)
        inside = 0.05 * normalized.pow(2).mean(dim=-1)
        cost = outside + inside
        valid = self.part_prior_valid_mask.to(points.device)
        return torch.where(valid[None], cost, torch.zeros_like(cost))

    @torch.no_grad()
    def record_reassignment_attempt(
        self,
        indices: Tensor,
        current_labels: Tensor,
        best_labels: Tensor,
        global_step: int,
    ):
        if indices.numel() == 0:
            return
        indices = indices.to(self.reassign_attempt_count.device).long()
        self.reassign_attempt_count[indices] += 1
        self.reassign_last_step[indices] = int(global_step)
        self.reassign_last_from_label[indices] = current_labels.to(
            self.reassign_last_from_label.device
        ).long()
        self.reassign_last_to_label[indices] = best_labels.to(
            self.reassign_last_to_label.device
        ).long()

    @torch.no_grad()
    def set_hard_assignments(
        self,
        indices: Tensor,
        labels: Tensor,
        logit_strength: float,
        reset_ghost_stats: bool = True,
    ):
        if indices.numel() == 0:
            return
        logits = self.fg.params["motion_coefs"]
        indices = indices.to(logits.device).long()
        labels = labels.to(logits.device).long()
        new_logits = torch.full(
            (indices.shape[0], logits.shape[1]),
            -float(logit_strength),
            device=logits.device,
            dtype=logits.dtype,
        )
        new_logits.scatter_(1, labels[:, None], float(logit_strength))
        logits.data[indices] = new_logits
        if self.initial_motion_logits.shape == logits.shape:
            self.initial_motion_logits[indices] = new_logits
        if self.assignment_target_labels.shape[0] == logits.shape[0]:
            self.assignment_target_labels[indices] = labels.to(
                self.assignment_target_labels.device
            )
        if self.assignment_spatial_target_labels.shape[0] == logits.shape[0]:
            self.assignment_spatial_target_labels[indices] = labels.to(
                self.assignment_spatial_target_labels.device
            )
        self.assignment_train_mask[indices.to(self.assignment_train_mask.device)] = True
        self.opacity_train_mask[indices.to(self.opacity_train_mask.device)] = True
        self.means_train_mask[indices.to(self.means_train_mask.device)] = True
        if reset_ghost_stats:
            for name in [
                "ghost_visible_count",
                "ghost_outside_count",
                "ghost_depth_bad_count",
                "ghost_score",
            ]:
                buffer = getattr(self, name)
                buffer[indices.to(buffer.device)] = 0

    def cull_foreground_gaussians(self, should_cull: Tensor) -> dict[str, nn.Parameter]:
        if should_cull.shape != (self.num_fg_gaussians,):
            raise ValueError(
                f"should_cull shape {tuple(should_cull.shape)} does not match "
                f"num_fg_gaussians={self.num_fg_gaussians}"
            )
        should_cull = should_cull.to(device=self.fg.params["means"].device).bool()
        keep = ~should_cull
        old_num_fg = int(should_cull.shape[0])
        fg_param_map = self.fg.cull_params(should_cull)

        def _filter_buffer(name: str):
            tensor = getattr(self, name)
            if tensor.shape[0] == old_num_fg:
                setattr(self, name, tensor[keep])

        for name in [
            "initial_motion_logits",
            "assignment_target_labels",
            "assignment_spatial_target_labels",
            "assignment_spatial_valid_mask",
            "initial_fg_means",
            "initial_fg_opacities",
            "opacity_train_mask",
            "means_train_mask",
            "assignment_train_mask",
            "ghost_visible_count",
            "ghost_outside_count",
            "ghost_depth_bad_count",
            "ghost_score",
            "reassign_attempt_count",
            "reassign_last_step",
            "reassign_last_from_label",
            "reassign_last_to_label",
        ]:
            _filter_buffer(name)

        if self.assignment_graph_edges.numel() > 0:
            old_to_new = torch.full(
                (old_num_fg,),
                -1,
                dtype=torch.long,
                device=self.assignment_graph_edges.device,
            )
            old_to_new[keep.to(old_to_new.device)] = torch.arange(
                int(keep.sum().item()),
                device=old_to_new.device,
                dtype=torch.long,
            )
            edges = self.assignment_graph_edges
            edge_keep = keep.to(edges.device)[edges[:, 0]] & keep.to(edges.device)[edges[:, 1]]
            self.assignment_graph_edges = old_to_new[edges[edge_keep]]
            self.assignment_graph_weights = self.assignment_graph_weights[edge_keep]

        self._register_assignment_grad_mask()
        self._register_gaussian_attr_grad_masks()
        return fg_param_map

    def hard_assignment_counts(self) -> Tensor:
        labels = self.fg.params["motion_coefs"].argmax(dim=-1)
        return torch.bincount(labels, minlength=self.num_motion_bases)

    @torch.no_grad()
    def get_joint_viz_segments(self, use_initial: bool = False) -> dict[str, Tensor]:
        return self.motion_bases.get_joint_viz_segments(use_initial=use_initial)

    @torch.no_grad()
    def get_gt_joint_viz_segments(self) -> dict[str, Tensor]:
        return self.motion_bases.get_gt_joint_viz_segments()

    def compute_transforms(self, ts: Tensor, inds: Tensor | None = None) -> Tensor:
        coefs = self.get_motion_coefs(inds=inds)
        return self.motion_bases.compute_transforms(ts, coefs)

    @staticmethod
    def init_from_stage2_state_dict(state_dict: dict[str, Tensor], prefix: str = ""):
        fg = GaussianParams.init_from_state_dict(state_dict, prefix=f"{prefix}fg.params.")
        bg = None
        if any("bg." in k for k in state_dict):
            bg = GaussianParams.init_from_state_dict(state_dict, prefix=f"{prefix}bg.params.")
        motion_bases = ArticulatedMotionBasesArticulat3DStage2.init_from_state_dict(
            state_dict, prefix=f"{prefix}motion_bases."
        )
        Ks = state_dict[f"{prefix}Ks"]
        w2cs = state_dict[f"{prefix}w2cs"]
        initial_logits = state_dict.get(f"{prefix}initial_motion_logits")
        assignment_train_mask = state_dict.get(f"{prefix}assignment_train_mask")
        assignment_target_labels = state_dict.get(f"{prefix}assignment_target_labels")
        assignment_spatial_target_labels = state_dict.get(
            f"{prefix}assignment_spatial_target_labels"
        )
        assignment_spatial_valid_mask = state_dict.get(f"{prefix}assignment_spatial_valid_mask")
        assignment_graph_edges = state_dict.get(f"{prefix}assignment_graph_edges")
        assignment_graph_weights = state_dict.get(f"{prefix}assignment_graph_weights")
        initial_fg_means = state_dict.get(f"{prefix}initial_fg_means")
        initial_fg_opacities = state_dict.get(f"{prefix}initial_fg_opacities")
        opacity_train_mask = state_dict.get(f"{prefix}opacity_train_mask")
        means_train_mask = state_dict.get(f"{prefix}means_train_mask")
        ghost_visible_count = state_dict.get(f"{prefix}ghost_visible_count")
        ghost_outside_count = state_dict.get(f"{prefix}ghost_outside_count")
        ghost_depth_bad_count = state_dict.get(f"{prefix}ghost_depth_bad_count")
        ghost_score = state_dict.get(f"{prefix}ghost_score")
        reassign_attempt_count = state_dict.get(f"{prefix}reassign_attempt_count")
        reassign_last_step = state_dict.get(f"{prefix}reassign_last_step")
        reassign_last_from_label = state_dict.get(f"{prefix}reassign_last_from_label")
        reassign_last_to_label = state_dict.get(f"{prefix}reassign_last_to_label")
        part_prior_centers = state_dict.get(f"{prefix}part_prior_centers")
        part_prior_axes = state_dict.get(f"{prefix}part_prior_axes")
        part_prior_scales = state_dict.get(f"{prefix}part_prior_scales")
        part_prior_valid_mask = state_dict.get(f"{prefix}part_prior_valid_mask")
        camera_poses = None
        if any("camera_poses." in k for k in state_dict):
            camera_poses = CameraPoses.init_from_state_dict(
                state_dict, prefix=f"{prefix}camera_poses.params."
            )
        return SceneModelArticulat3DStage2(
            Ks=Ks,
            w2cs=w2cs,
            fg_params=fg,
            motion_bases=motion_bases,
            camera_poses=camera_poses,
            bg_params=bg,
            initial_motion_logits=initial_logits,
            assignment_train_mask=assignment_train_mask,
            assignment_target_labels=assignment_target_labels,
            assignment_spatial_target_labels=assignment_spatial_target_labels,
            assignment_spatial_valid_mask=assignment_spatial_valid_mask,
            assignment_graph_edges=assignment_graph_edges,
            assignment_graph_weights=assignment_graph_weights,
            initial_fg_means=initial_fg_means,
            initial_fg_opacities=initial_fg_opacities,
            opacity_train_mask=opacity_train_mask,
            means_train_mask=means_train_mask,
            ghost_visible_count=ghost_visible_count,
            ghost_outside_count=ghost_outside_count,
            ghost_depth_bad_count=ghost_depth_bad_count,
            ghost_score=ghost_score,
            reassign_attempt_count=reassign_attempt_count,
            reassign_last_step=reassign_last_step,
            reassign_last_from_label=reassign_last_from_label,
            reassign_last_to_label=reassign_last_to_label,
            part_prior_centers=part_prior_centers,
            part_prior_axes=part_prior_axes,
            part_prior_scales=part_prior_scales,
            part_prior_valid_mask=part_prior_valid_mask,
        )
