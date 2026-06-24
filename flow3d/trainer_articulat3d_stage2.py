from __future__ import annotations

import functools
import time
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from loguru import logger as guru

from flow3d.articulated_motion_articulat3d_stage2 import SceneModelArticulat3DStage2
from flow3d.configs import LossesConfig, OptimizerConfig, SceneLRConfig
from flow3d.loss_utils import compute_gradient_loss, compute_z_acc_loss, masked_l1_loss
from flow3d.trainer_articulat3d import TrainerArticulat3D, disable_lpips_metric_downloads


class TrainerArticulat3DStage2(TrainerArticulat3D):
    def __init__(
        self,
        model: SceneModelArticulat3DStage2,
        device: torch.device,
        lr_cfg: SceneLRConfig,
        losses_cfg: LossesConfig,
        optim_cfg: OptimizerConfig,
        work_dir: str,
        port: int | None = None,
        motion_segment_length: int = 0,
        joint_value_lr: float = 1.6e-4,
        joint_axis_lr: float = 1.0e-5,
        joint_pivot_lr: float = 1.0e-5,
        unlock_joint_geometry_after_steps: int = 1000,
        assignment_tau_init: float = 1.0,
        assignment_tau_final: float = 0.1,
        assignment_tau_decay_steps: int = 5000,
        use_assignment_pseudo_targets: bool = False,
        w_assignment_prior: float = 0.0,
        w_assignment_spatial_prior: float = 0.0,
        w_assignment_graph_smooth: float = 0.0,
        w_assignment_entropy: float = 0.0,
        use_stochastic_assignment: bool = False,
        w_axis_prior: float = 1.0e-3,
        w_pivot_prior: float = 1.0e-3,
        unlock_all_assignment_epoch: int = 50,
        unlock_opacity_epoch: int = 50,
        unlock_means_epoch: int = 100,
        stage2_opacity_lr: float = 1.0e-3,
        stage2_means_lr: float = 1.0e-5,
        w_opacity_anchor: float = 1.0e-4,
        w_means_anchor: float = 1.0e-2,
        train_stage2_colors: bool = False,
        train_stage2_quats: bool = False,
        train_stage2_scales: bool = False,
        disable_gaussian_control: bool = True,
        w_outside_alpha: float = 0.5,
        outside_mask_dilate_px: int = 3,
        enable_ghost_score: bool = True,
        ghost_start_epoch: int = 50,
        ghost_score_update_every_steps: int = 20,
        ghost_min_visible: int = 5,
        ghost_depth_abs_thresh: float = 0.05,
        ghost_depth_rel_thresh: float = 0.03,
        ghost_opacity_score_thresh: float = 0.6,
        w_ghost_opacity_decay: float = 5.0e-3,
        enable_stage2_ghost_cull: bool = True,
        ghost_cull_every_steps: int = 500,
        ghost_cull_score_thresh: float = 0.75,
        ghost_cull_opacity_thresh: float = 0.08,
        max_ghost_cull_per_step: int = 2000,
        enable_direct_reassignment: bool = True,
        assignment_reassign_start_epoch: int = 50,
        assignment_reassign_every_steps: int = 500,
        max_reassign_per_step: int = 3000,
        reassign_candidate_score_thresh: float = 0.6,
        reassign_min_visible: int = 5,
        reassign_min_opacity: float = 0.05,
        reassign_logit_strength: float = 4.0,
        reassign_improve_ratio: float = 0.20,
        w_reassign_mask: float = 1.0,
        w_reassign_depth: float = 2.0,
        w_reassign_part_prior: float = 0.0,
        w_reassign_knn: float = 0.1,
        reassign_switch_penalty: float = 0.05,
        part_prior_min_points: int = 128,
        part_prior_conf_thresh: float = -1.0,
        part_prior_opacity_thresh: float = 0.1,
        part_prior_ghost_score_thresh: float = 0.5,
        part_prior_scale_min: float = 0.03,
        part_prior_center_mode: str = "robust_obb",
        part_prior_obb_low_quantile: float = 0.05,
        part_prior_obb_high_quantile: float = 0.95,
        ghost_cull_requires_reassign_attempt: bool = True,
        show_joint_axes: bool = True,
        show_gt_joint_axes: bool = True,
        viewer_joint_update_every: int = 100,
        joint_axis_init_line_width: float = 2.0,
        joint_axis_current_line_width: float = 4.0,
        gt_joint_axis_line_width: float = 3.0,
        joint_pivot_point_size: float = 0.035,
        show_part_prior_centers: bool = False,
        part_prior_center_point_size: float = 0.08,
        viewer_part_centers_follow_motion: bool = True,
        viewer_render_uses_timestep: bool = True,
        freeze_gaussian_params: bool = True,
        train_axis_pivot: bool = False,
        **kwargs,
    ):
        self.joint_value_lr = joint_value_lr
        self.joint_axis_lr = joint_axis_lr
        self.joint_pivot_lr = joint_pivot_lr
        self.unlock_joint_geometry_after_steps = unlock_joint_geometry_after_steps
        self.assignment_tau_init = assignment_tau_init
        self.assignment_tau_final = assignment_tau_final
        self.assignment_tau_decay_steps = assignment_tau_decay_steps
        self.use_assignment_pseudo_targets = use_assignment_pseudo_targets
        self.use_stochastic_assignment = use_stochastic_assignment
        self.w_assignment_prior = w_assignment_prior
        self.w_assignment_spatial_prior = w_assignment_spatial_prior
        self.w_assignment_graph_smooth = w_assignment_graph_smooth
        self.w_assignment_entropy = w_assignment_entropy
        self.w_axis_prior = w_axis_prior
        self.w_pivot_prior = w_pivot_prior
        self.unlock_all_assignment_epoch = unlock_all_assignment_epoch
        self.unlock_opacity_epoch = unlock_opacity_epoch
        self.unlock_means_epoch = unlock_means_epoch
        self.stage2_opacity_lr = stage2_opacity_lr
        self.stage2_means_lr = stage2_means_lr
        self.w_opacity_anchor = w_opacity_anchor
        self.w_means_anchor = w_means_anchor
        self.train_stage2_colors = train_stage2_colors
        self.train_stage2_quats = train_stage2_quats
        self.train_stage2_scales = train_stage2_scales
        self.disable_gaussian_control = disable_gaussian_control
        self.w_outside_alpha = w_outside_alpha
        self.outside_mask_dilate_px = outside_mask_dilate_px
        self.enable_ghost_score = enable_ghost_score
        self.ghost_start_epoch = ghost_start_epoch
        self.ghost_score_update_every_steps = ghost_score_update_every_steps
        self.ghost_min_visible = ghost_min_visible
        self.ghost_depth_abs_thresh = ghost_depth_abs_thresh
        self.ghost_depth_rel_thresh = ghost_depth_rel_thresh
        self.ghost_opacity_score_thresh = ghost_opacity_score_thresh
        self.w_ghost_opacity_decay = w_ghost_opacity_decay
        self.enable_stage2_ghost_cull = enable_stage2_ghost_cull
        self.ghost_cull_every_steps = ghost_cull_every_steps
        self.ghost_cull_score_thresh = ghost_cull_score_thresh
        self.ghost_cull_opacity_thresh = ghost_cull_opacity_thresh
        self.max_ghost_cull_per_step = max_ghost_cull_per_step
        self.enable_direct_reassignment = enable_direct_reassignment
        self.assignment_reassign_start_epoch = assignment_reassign_start_epoch
        self.assignment_reassign_every_steps = assignment_reassign_every_steps
        self.max_reassign_per_step = max_reassign_per_step
        self.reassign_candidate_score_thresh = reassign_candidate_score_thresh
        self.reassign_min_visible = reassign_min_visible
        self.reassign_min_opacity = reassign_min_opacity
        self.reassign_logit_strength = reassign_logit_strength
        self.reassign_improve_ratio = reassign_improve_ratio
        self.w_reassign_mask = w_reassign_mask
        self.w_reassign_depth = w_reassign_depth
        self.w_reassign_part_prior = w_reassign_part_prior
        self.w_reassign_knn = w_reassign_knn
        self.reassign_switch_penalty = reassign_switch_penalty
        self.part_prior_min_points = part_prior_min_points
        self.part_prior_conf_thresh = part_prior_conf_thresh
        self.part_prior_opacity_thresh = part_prior_opacity_thresh
        self.part_prior_ghost_score_thresh = part_prior_ghost_score_thresh
        self.part_prior_scale_min = part_prior_scale_min
        self.part_prior_center_mode = part_prior_center_mode
        self.part_prior_obb_low_quantile = part_prior_obb_low_quantile
        self.part_prior_obb_high_quantile = part_prior_obb_high_quantile
        self.ghost_cull_requires_reassign_attempt = ghost_cull_requires_reassign_attempt
        self.show_joint_axes = show_joint_axes
        self.show_gt_joint_axes = show_gt_joint_axes
        self.viewer_joint_update_every = viewer_joint_update_every
        self.joint_axis_init_line_width = joint_axis_init_line_width
        self.joint_axis_current_line_width = joint_axis_current_line_width
        self.gt_joint_axis_line_width = gt_joint_axis_line_width
        self.joint_pivot_point_size = joint_pivot_point_size
        self.show_part_prior_centers = show_part_prior_centers
        self.part_prior_center_point_size = part_prior_center_point_size
        self.viewer_part_centers_follow_motion = viewer_part_centers_follow_motion
        self.viewer_render_uses_timestep = viewer_render_uses_timestep
        self.freeze_gaussian_params = freeze_gaussian_params
        self.train_axis_pivot = train_axis_pivot
        self._joint_axis_handles = {}
        self._part_prior_center_timestep_callback_registered = False
        self._last_reassign_batch = None
        self._last_reassignment_stats = {
            "candidate_count": 0.0,
            "switched_count": 0.0,
            "mean_cost_improve": 0.0,
            "ghost_protected_count": 0.0,
            "ghost_culled_count": 0.0,
        }
        self._apply_stage2_train_policy(model)
        disable_lpips_metric_downloads()
        super().__init__(
            model=model,
            device=device,
            lr_cfg=lr_cfg,
            losses_cfg=losses_cfg,
            optim_cfg=optim_cfg,
            work_dir=work_dir,
            port=port,
            motion_segment_length=motion_segment_length,
            **kwargs,
        )
        self.model.set_stochastic_assignment(self.use_stochastic_assignment)
        if self.show_joint_axes:
            self._setup_joint_axis_viewer()
        if self.show_part_prior_centers:
            self._setup_part_prior_center_viewer()

    @staticmethod
    def init_from_checkpoint(
        path: str, device: torch.device, use_2dgs, *args, **kwargs
    ) -> tuple["TrainerArticulat3DStage2", int]:
        guru.info(f"Loading stage2 checkpoint from {path}")
        ckpt = torch.load(path, weights_only=False, map_location=device)
        model = SceneModelArticulat3DStage2.init_from_stage2_state_dict(ckpt["model"])
        model = model.to(device)
        model.use_2dgs = use_2dgs
        trainer = TrainerArticulat3DStage2(model, device, *args, **kwargs)
        if "optimizers" in ckpt:
            trainer.load_checkpoint_optimizers(ckpt["optimizers"])
        if "schedulers" in ckpt:
            trainer.load_checkpoint_schedulers(ckpt["schedulers"])
        trainer.global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0)
        trainer.set_epoch(start_epoch)
        return trainer, start_epoch

    def _assignment_tau(self) -> float:
        if self.assignment_tau_decay_steps <= 0:
            return self.assignment_tau_final
        alpha = min(float(self.global_step) / float(self.assignment_tau_decay_steps), 1.0)
        if self.assignment_tau_init <= 0 or self.assignment_tau_final <= 0:
            return self.assignment_tau_final
        log_tau = (1.0 - alpha) * np.log(self.assignment_tau_init) + alpha * np.log(
            self.assignment_tau_final
        )
        return float(np.exp(log_tau))

    def load_checkpoint_optimizers(self, opt_ckpt):
        for k, v in self.optimizers.items():
            if k in opt_ckpt:
                v.load_state_dict(opt_ckpt[k])
            else:
                guru.warning(f"Skipping missing optimizer state for new stage2 param: {k}")

    def load_checkpoint_schedulers(self, sched_ckpt):
        for k, v in self.scheduler.items():
            if k in sched_ckpt:
                v.load_state_dict(sched_ckpt[k])
            else:
                guru.warning(f"Skipping missing scheduler state for new stage2 param: {k}")

    def train_step(self, batch):
        self._update_stage2_train_phase()
        self.model.set_assignment_temperature(self._assignment_tau())
        loss = super().train_step(batch)
        if (
            (self.show_joint_axes or self.show_part_prior_centers)
            and self.viewer is not None
            and self.viewer_joint_update_every > 0
            and self.global_step % self.viewer_joint_update_every == 0
        ):
            if self.show_joint_axes:
                self._update_current_joint_axis_viewer()
            if self.show_part_prior_centers:
                self._update_part_prior_center_viewer()
        return loss

    def run_control_steps(self):
        self._run_stage2_direct_reassignment()
        self._run_stage2_ghost_cull()
        if self.disable_gaussian_control:
            return
        return super().run_control_steps()

    def configure_optimizers(self):
        lr_dict = asdict(self.lr_cfg)
        optimizers = {}
        schedulers = {}

        def lr_for_param(name: str) -> float | None:
            if name.startswith("fg.params."):
                field = name.split(".")[-1]
                if field == "motion_coefs":
                    return lr_dict["fg"][field]
                if field == "opacities":
                    return self.stage2_opacity_lr
                if field == "means":
                    return self.stage2_means_lr
                if field == "colors" and self.train_stage2_colors:
                    return lr_dict["fg"][field]
                if field == "quats" and self.train_stage2_quats:
                    return lr_dict["fg"][field]
                if field == "scales" and self.train_stage2_scales:
                    return lr_dict["fg"][field]
                return None
            if name.startswith("bg.params."):
                return None
            if name == "motion_bases.params.joint_values":
                return self.joint_value_lr
            if name == "motion_bases.params.axis_raw":
                if not self.train_axis_pivot:
                    return None
                return self.joint_axis_lr
            if name == "motion_bases.params.pivot":
                if not self.train_axis_pivot:
                    return None
                return self.joint_pivot_lr
            if name.startswith("camera_poses.params."):
                return lr_dict["camera_poses"][name.split(".")[-1]]
            return None

        def schedule_factor(name: str, step: int) -> float:
            epoch = getattr(self, "epoch", 0)
            if name == "fg.params.opacities":
                return 1.0 if epoch >= self.unlock_opacity_epoch else 0.0
            if name == "fg.params.means":
                return 1.0 if epoch >= self.unlock_means_epoch else 0.0
            if name in {"motion_bases.params.axis_raw", "motion_bases.params.pivot"}:
                return 0.0 if step < self.unlock_joint_geometry_after_steps else 1.0
            if "scales" in name:
                return np.exp(np.log(1.0) * (1 - min(step / self.optim_cfg.max_steps, 1.0)) + np.log(0.1) * min(step / self.optim_cfg.max_steps, 1.0))
            return 1.0

        for name, params in self.model.named_parameters():
            if not params.requires_grad:
                continue
            lr = lr_for_param(name)
            if lr is None:
                continue
            optim = torch.optim.Adam([{"params": params, "lr": lr, "name": name}])
            schedulers[name] = torch.optim.lr_scheduler.LambdaLR(
                optim,
                functools.partial(schedule_factor, name),
            )
            optimizers[name] = optim
        return optimizers, schedulers

    def compute_losses(self, batch):
        self.model.training = True

        B = batch["imgs"].shape[0]
        W, H = img_wh = batch["imgs"].shape[2:0:-1]
        ts = batch["ts"]
        w2cs = batch["w2cs"]
        Ks = batch["Ks"]
        imgs = batch["imgs"]
        valid_masks = batch.get("valid_masks", torch.ones_like(batch["imgs"][..., 0]))
        masks = batch["masks"]
        masks *= valid_masks
        depths = batch["depths"]
        dilated_masks = self._dilate_mask(masks, self.outside_mask_dilate_px)
        self._last_reassign_batch = {
            "ts": ts.detach(),
            "w2cs": w2cs.detach(),
            "Ks": Ks.detach(),
            "masks": masks.detach(),
            "dilated_masks": dilated_masks.detach(),
            "depths": depths.detach(),
        }

        _tic = time.time()
        means, quats = self.model.compute_poses_all(ts)
        device = means.device
        means = means.transpose(0, 1)
        quats = quats.transpose(0, 1)
        num_frames = self.model.num_frames

        loss = 0.0
        bg_colors = []
        rendered_all = []
        self._batched_xys = []
        self._batched_radii = []
        self._batched_img_wh = []
        for i in range(B):
            bg_color = torch.ones(1, 3, device=device)
            rendered = self.model.render(
                ts[i].item(),
                w2cs[None, i],
                Ks[None, i],
                img_wh,
                bg_color=bg_color,
                means=means[i],
                quats=quats[i],
                return_depth=True,
                return_mask=self.model.has_bg,
            )
            rendered_all.append(rendered)
            bg_colors.append(bg_color)
            if (
                self.model._current_xys is not None
                and self.model._current_radii is not None
                and self.model._current_img_wh is not None
            ):
                self._batched_xys.append(self.model._current_xys)
                self._batched_radii.append(self.model._current_radii)
                self._batched_img_wh.append(self.model._current_img_wh)

        num_rays_per_step = H * W * B
        num_rays_per_sec = num_rays_per_step / (time.time() - _tic)

        rendered_all = {
            key: (
                torch.cat([out_dict[key] for out_dict in rendered_all], dim=0)
                if rendered_all[0][key] is not None
                else None
            )
            for key in rendered_all[0]
        }
        bg_colors = torch.cat(bg_colors, dim=0)

        if not self.model.has_bg:
            imgs = imgs * masks[..., None] + (1.0 - masks[..., None]) * bg_colors[:, None, None]
        else:
            imgs = imgs * valid_masks[..., None] + (1.0 - valid_masks[..., None]) * bg_colors[:, None, None]

        if rendered_all["rend_normal"] != None and rendered_all["surf_normal"] != None:
            rendered_normals = cast(torch.Tensor, rendered_all["rend_normal"])
            surf_normals = cast(torch.Tensor, rendered_all["surf_normal"]).reshape(rendered_normals.shape)
            loss += (1 - torch.sum(rendered_normals * surf_normals, dim=-1)).mean() * 0.05

        rendered_imgs = cast(torch.Tensor, rendered_all["img"])
        if self.model.has_bg:
            rendered_imgs = rendered_imgs * valid_masks[..., None] + (
                1.0 - valid_masks[..., None]
            ) * bg_colors[:, None, None]
        rgb_loss = 0.8 * F.l1_loss(rendered_imgs, imgs) + 0.2 * (
            1 - self.ssim(rendered_imgs.permute(0, 3, 1, 2), imgs.permute(0, 3, 1, 2))
        )
        loss += rgb_loss * self.losses_cfg.w_rgb

        if not self.model.has_bg:
            mask_loss = F.mse_loss(rendered_all["acc"], masks[..., None])  # type: ignore[arg-type]
        else:
            mask_loss = F.mse_loss(
                rendered_all["acc"], torch.ones_like(rendered_all["acc"])  # type: ignore[arg-type]
            ) + masked_l1_loss(rendered_all["mask"], masks[..., None], quantile=0.98)
        loss += mask_loss * self.losses_cfg.w_mask

        outside_mask = (valid_masks > 0.5) & ~(dilated_masks > 0.5)
        outside_alpha_src = (
            rendered_all["mask"]
            if self.model.has_bg and rendered_all.get("mask") is not None
            else rendered_all["acc"]
        )
        outside_alpha_loss = self._masked_mean(
            cast(torch.Tensor, outside_alpha_src).abs(),
            outside_mask[..., None],
        )
        loss += outside_alpha_loss * self.w_outside_alpha

        depth_masks = masks[..., None] if not self.model.has_bg else valid_masks[..., None]
        pred_depth = cast(torch.Tensor, rendered_all["depth"])
        pred_disp = 1.0 / (pred_depth + 1e-5)
        tgt_disp = 1.0 / (depths[..., None] + 1e-5)
        depth_loss = masked_l1_loss(pred_disp, tgt_disp, mask=depth_masks, quantile=0.98)
        loss += depth_loss * self.losses_cfg.w_depth_reg

        depth_gradient_loss = compute_gradient_loss(
            pred_disp,
            tgt_disp,
            mask=depth_masks > 0.5,
            quantile=0.95,
        )
        loss += depth_gradient_loss * self.losses_cfg.w_depth_grad

        joint_value_smoothness = self.model.motion_bases.value_smoothness_loss(
            self.motion_segment_length
        )
        loss += joint_value_smoothness * self.losses_cfg.w_smooth_bases

        assignment_prior_loss = torch.zeros((), device=device)
        assignment_spatial_prior_loss = torch.zeros((), device=device)
        assignment_graph_smooth_loss = self.model.assignment_graph_smooth_loss()
        if self.epoch >= self.unlock_all_assignment_epoch:
            assignment_entropy_loss = self.model.assignment_entropy_loss()
        else:
            assignment_entropy_loss = torch.zeros((), device=device)
        if self.epoch >= self.unlock_opacity_epoch:
            opacity_anchor_loss = self.model.opacity_anchor_loss()
        else:
            opacity_anchor_loss = torch.zeros((), device=device)
        if self.epoch >= self.unlock_means_epoch:
            means_anchor_loss = self.model.means_anchor_loss()
        else:
            means_anchor_loss = torch.zeros((), device=device)
        if (
            self.enable_ghost_score
            and self.epoch >= self.ghost_start_epoch
            and self.ghost_score_update_every_steps > 0
            and self.global_step % self.ghost_score_update_every_steps == 0
        ):
            self.model.update_ghost_scores(
                means_fg=means[:, : self.model.num_fg_gaussians].detach(),
                w2cs=w2cs.detach(),
                Ks=Ks.detach(),
                masks=masks.detach(),
                dilated_masks=dilated_masks.detach(),
                depths=depths.detach(),
                depth_abs_thresh=self.ghost_depth_abs_thresh,
                depth_rel_thresh=self.ghost_depth_rel_thresh,
                score_thresh_for_opacity=self.ghost_opacity_score_thresh,
            )
        if self.enable_ghost_score and self.epoch >= self.ghost_start_epoch:
            ghost_opacity_decay_loss = self.model.ghost_opacity_decay_loss(
                score_thresh=self.ghost_opacity_score_thresh,
                min_visible=self.ghost_min_visible,
            )
        else:
            ghost_opacity_decay_loss = torch.zeros((), device=device)
        axis_prior_loss = self.model.motion_bases.axis_prior_loss()
        pivot_prior_loss = self.model.motion_bases.pivot_prior_loss()
        loss += assignment_prior_loss * self.w_assignment_prior
        loss += assignment_spatial_prior_loss * self.w_assignment_spatial_prior
        loss += assignment_graph_smooth_loss * self.w_assignment_graph_smooth
        loss += assignment_entropy_loss * self.w_assignment_entropy
        loss += opacity_anchor_loss * self.w_opacity_anchor
        loss += means_anchor_loss * self.w_means_anchor
        loss += ghost_opacity_decay_loss * self.w_ghost_opacity_decay
        loss += axis_prior_loss * self.w_axis_prior
        loss += pivot_prior_loss * self.w_pivot_prior

        interior_mask = self._interior_ts_mask(ts, num_frames)
        if interior_mask.any():
            ts_mid = ts[interior_mask]
            ts_neighbors = torch.cat((ts_mid - 1, ts_mid, ts_mid + 1))
            transfms_nbs = self.model.compute_transforms(ts_neighbors)
            means_fg_nbs = torch.einsum(
                "pnij,pj->pni",
                transfms_nbs,
                F.pad(self.model.fg.params["means"], (0, 1), value=1.0),
            )
            means_fg_nbs = means_fg_nbs.reshape(means_fg_nbs.shape[0], 3, -1, 3)
            if self.losses_cfg.w_smooth_tracks > 0:
                small_accel_loss_tracks = 0.5 * (
                    (2 * means_fg_nbs[:, 1:-1] - means_fg_nbs[:, :-2] - means_fg_nbs[:, 2:])
                    .norm(dim=-1)
                    .mean()
                )
                loss += small_accel_loss_tracks * self.losses_cfg.w_smooth_tracks
            else:
                small_accel_loss_tracks = torch.zeros((), device=device)
            z_accel_loss = compute_z_acc_loss(means_fg_nbs, w2cs[interior_mask])
            loss += self.losses_cfg.w_z_accel * z_accel_loss
        else:
            small_accel_loss_tracks = torch.zeros((), device=device)
            z_accel_loss = torch.zeros((), device=device)

        loss += (
            self.losses_cfg.w_scale_var
            * torch.var(torch.exp(self.model.fg.params["scales"]), dim=-1).mean()
        )
        if self.model.bg is not None:
            loss += (
                self.losses_cfg.w_scale_var
                * torch.var(torch.exp(self.model.bg.params["scales"]), dim=-1).mean()
            )

        counts = self.model.hard_assignment_counts().detach()
        stats = {
            "train/loss": loss.item(),
            "train/rgb_loss": rgb_loss.item(),
            "train/mask_loss": mask_loss.item(),
            "train/outside_alpha_loss": outside_alpha_loss.item(),
            "train/depth_loss": depth_loss.item(),
            "train/depth_gradient_loss": depth_gradient_loss.item(),
            "train/joint_value_smoothness": joint_value_smoothness.item(),
            "train/small_accel_loss_tracks": small_accel_loss_tracks.item(),
            "train/z_acc_loss": z_accel_loss.item(),
            "train/assignment_prior_loss": assignment_prior_loss.item(),
            "train/assignment_spatial_prior_loss": assignment_spatial_prior_loss.item(),
            "train/assignment_graph_smooth_loss": assignment_graph_smooth_loss.item(),
            "train/assignment_entropy_loss": assignment_entropy_loss.item(),
            "train/opacity_anchor_loss": opacity_anchor_loss.item(),
            "train/means_anchor_loss": means_anchor_loss.item(),
            "train/ghost_opacity_decay_loss": ghost_opacity_decay_loss.item(),
            "train/axis_prior_loss": axis_prior_loss.item(),
            "train/pivot_prior_loss": pivot_prior_loss.item(),
            "train/assignment_tau": self.model.assignment_tau,
            "train/joint_geometry_unlocked": float(
                self.train_axis_pivot
                and self.global_step >= self.unlock_joint_geometry_after_steps
            ),
            "train/assignment_trainable_count": float(
                self.model.assignment_train_mask.sum().item()
            ),
            "train/assignment_refine_trainable_count": float(
                self.model.assignment_train_mask.sum().item()
            ),
            "train/assignment_spatial_valid_count": float(
                self.model.assignment_spatial_valid_mask.sum().item()
            ),
            "train/assignment_graph_edge_count": float(
                self.model.assignment_graph_edges.shape[0]
            ),
            "train/unlocked_assignment_all": float(
                self.epoch >= self.unlock_all_assignment_epoch
            ),
            "train/unlocked_opacity": float(self.epoch >= self.unlock_opacity_epoch),
            "train/unlocked_means": float(self.epoch >= self.unlock_means_epoch),
            "train/trainable_opacity_count": float(
                self.model.opacity_train_mask.sum().item()
                if self.epoch >= self.unlock_opacity_epoch
                else 0
            ),
            "train/trainable_means_count": float(
                self.model.means_train_mask.sum().item()
                if self.epoch >= self.unlock_means_epoch
                else 0
            ),
            "train/use_assignment_pseudo_targets": float(self.use_assignment_pseudo_targets),
            "train/freeze_gaussian_params": float(self.freeze_gaussian_params),
            "train/train_axis_pivot": float(self.train_axis_pivot),
            "train/ghost_mean_score": float(self.model.ghost_score.mean().item()),
            "train/ghost_candidate_count": float(
                (
                    (self.model.ghost_score >= self.ghost_opacity_score_thresh)
                    & (self.model.ghost_visible_count >= self.ghost_min_visible)
                )
                .sum()
                .item()
            ),
            "train/reassign_candidate_count": self._last_reassignment_stats[
                "candidate_count"
            ],
            "train/reassign_switched_count": self._last_reassignment_stats[
                "switched_count"
            ],
            "train/reassign_mean_cost_improve": self._last_reassignment_stats[
                "mean_cost_improve"
            ],
            "train/ghost_protected_count": self._last_reassignment_stats[
                "ghost_protected_count"
            ],
            "train/ghost_culled_count": self._last_reassignment_stats[
                "ghost_culled_count"
            ],
            "train/num_gaussians": self.model.num_gaussians,
            "train/num_fg_gaussians": self.model.num_fg_gaussians,
            "train/num_bg_gaussians": self.model.num_bg_gaussians,
        }
        for i, count in enumerate(counts.tolist()):
            stats[f"train/part_{i}_count"] = float(count)

        with torch.no_grad():
            psnr = self.psnr_metric(
                rendered_imgs, imgs, masks if not self.model.has_bg else valid_masks
            )
            self.psnr_metric.reset()
            stats["train/psnr"] = psnr

        stats.update(
            **{
                "train/num_rays_per_sec": num_rays_per_sec,
                "train/num_rays_per_step": float(num_rays_per_step),
            }
        )
        return loss, stats, num_rays_per_step, num_rays_per_sec

    @staticmethod
    def _dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
        mask = mask.float()
        if radius <= 0:
            return mask
        kernel = 2 * int(radius) + 1
        return F.max_pool2d(mask[:, None], kernel_size=kernel, stride=1, padding=radius)[:, 0]

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=values.device, dtype=values.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (values * mask).sum() / denom

    def _apply_stage2_train_policy(self, model: SceneModelArticulat3DStage2):
        for name, param in model.fg.params.items():
            param.requires_grad_(name in {"motion_coefs", "opacities", "means"})
            if name == "colors":
                param.requires_grad_(self.train_stage2_colors)
            elif name == "quats":
                param.requires_grad_(self.train_stage2_quats)
            elif name == "scales":
                param.requires_grad_(self.train_stage2_scales)
        if model.bg is not None:
            for param in model.bg.params.values():
                param.requires_grad_(False)
        if not self.train_axis_pivot:
            model.motion_bases.params["axis_raw"].requires_grad_(False)
            model.motion_bases.params["pivot"].requires_grad_(False)

    def _update_stage2_train_phase(self):
        self.model.set_assignment_grad_mask_active(
            self.epoch < self.unlock_all_assignment_epoch
        )
        self.model.set_gaussian_attr_grad_enabled(
            means=self.epoch >= self.unlock_means_epoch,
            opacities=self.epoch >= self.unlock_opacity_epoch,
        )

    def _replace_param_in_optimizer_after_cull(
        self,
        optimizer: torch.optim.Optimizer,
        new_param: torch.nn.Parameter,
        should_cull: torch.Tensor,
    ):
        old_param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state.pop(old_param, {})
        if len(param_state) > 0:
            for key, value in list(param_state.items()):
                if key == "step":
                    continue
                param_state[key] = value[~should_cull]
            optimizer.state[new_param] = param_state
        optimizer.param_groups[0]["params"] = [new_param]

    @torch.no_grad()
    def _zero_optimizer_state_rows(self, optimizer_name: str, rows: torch.Tensor):
        optimizer = self.optimizers.get(optimizer_name)
        if optimizer is None or rows.numel() == 0:
            return
        param = optimizer.param_groups[0]["params"][0]
        state = optimizer.state.get(param, {})
        rows = rows.to(param.device).long()
        for key, value in state.items():
            if key == "step" or not torch.is_tensor(value):
                continue
            if value.shape[:1] == param.shape[:1]:
                value[rows] = 0

    @torch.no_grad()
    def _run_stage2_direct_reassignment(self):
        if (
            not self.enable_direct_reassignment
            or not self.enable_ghost_score
            or self.epoch < self.assignment_reassign_start_epoch
            or self.assignment_reassign_every_steps <= 0
            or self.global_step == 0
            or self.global_step % self.assignment_reassign_every_steps != 0
            or self._last_reassign_batch is None
        ):
            return

        score = self.model.ghost_score
        visible = self.model.ghost_visible_count
        opacity = self.model.fg.get_opacities().reshape(
            self.model.num_fg_gaussians, -1
        ).mean(dim=-1)
        high_ghost = (
            (score >= self.reassign_candidate_score_thresh)
            & (visible >= self.reassign_min_visible)
        )
        if not high_ghost.any():
            self._last_reassignment_stats = {
                "candidate_count": 0.0,
                "switched_count": 0.0,
                "mean_cost_improve": 0.0,
                "ghost_protected_count": 0.0,
                "ghost_culled_count": self._last_reassignment_stats.get(
                    "ghost_culled_count", 0.0
                ),
            }
            return

        logits = self.model.fg.params["motion_coefs"].detach()
        current_labels_all = logits.argmax(dim=-1)
        attempt_indices = torch.where(high_ghost)[0]
        self.model.record_reassignment_attempt(
            attempt_indices,
            current_labels_all[attempt_indices],
            current_labels_all[attempt_indices],
            self.global_step,
        )

        candidate_mask = high_ghost & (opacity >= self.reassign_min_opacity)
        candidate_indices = torch.where(candidate_mask)[0]
        if candidate_indices.numel() > self.max_reassign_per_step:
            top = torch.topk(
                score[candidate_indices],
                k=int(self.max_reassign_per_step),
                largest=True,
            ).indices
            candidate_indices = candidate_indices[top]

        prior_report = {
            "enabled": False,
            "reason": "part_center_prior_disabled",
        }

        switched_indices = torch.empty(0, device=score.device, dtype=torch.long)
        switched_from = torch.empty(0, device=score.device, dtype=torch.long)
        switched_to = torch.empty(0, device=score.device, dtype=torch.long)
        improve = torch.empty(0, device=score.device, dtype=score.dtype)
        cost_report = {}
        if candidate_indices.numel() > 0:
            costs = self._score_reassignment_candidates(candidate_indices)
            current_labels = current_labels_all[candidate_indices]
            row = torch.arange(candidate_indices.shape[0], device=costs.device)
            current_cost = costs[row, current_labels]
            best_cost, best_labels = costs.min(dim=-1)
            should_switch = (
                (best_labels != current_labels)
                & (best_cost <= current_cost * (1.0 - self.reassign_improve_ratio))
            )
            if should_switch.any():
                switched_indices = candidate_indices[should_switch]
                switched_from = current_labels[should_switch]
                switched_to = best_labels[should_switch]
                improve = current_cost[should_switch] - best_cost[should_switch]
                self.model.record_reassignment_attempt(
                    switched_indices,
                    switched_from,
                    switched_to,
                    self.global_step,
                )
                self.model.set_hard_assignments(
                    switched_indices,
                    switched_to,
                    logit_strength=self.reassign_logit_strength,
                    reset_ghost_stats=True,
                )
                self._zero_optimizer_state_rows("fg.params.motion_coefs", switched_indices)

            cost_report = {
                "current_cost_mean": float(current_cost.mean().item()),
                "best_cost_mean": float(best_cost.mean().item()),
                "current_cost_q50": float(torch.quantile(current_cost, 0.50).item()),
                "best_cost_q50": float(torch.quantile(best_cost, 0.50).item()),
            }

        protected = high_ghost & (opacity >= self.ghost_cull_opacity_thresh)
        stats = {
            "candidate_count": float(candidate_indices.numel()),
            "attempt_count": float(attempt_indices.numel()),
            "switched_count": float(switched_indices.numel()),
            "mean_cost_improve": float(improve.mean().item()) if improve.numel() else 0.0,
            "ghost_protected_count": float(protected.sum().item()),
            "ghost_culled_count": self._last_reassignment_stats.get(
                "ghost_culled_count", 0.0
            ),
        }
        self._last_reassignment_stats = stats
        for key, value in stats.items():
            self.writer.add_scalar(f"train/{key if key.startswith('ghost') else 'reassign_' + key}", value, self.global_step)
        self._write_reassignment_report(
            candidate_indices=candidate_indices,
            switched_from=switched_from,
            switched_to=switched_to,
            cost_report=cost_report,
            prior_report=prior_report,
            protected_count=int(protected.sum().item()),
        )

        if switched_indices.numel() > 0:
            guru.info(
                "Stage2 reassigned "
                f"{int(switched_indices.numel())}/{int(candidate_indices.numel())} "
                "high-ghost gaussians to new motion bases"
            )

    @torch.no_grad()
    def _score_reassignment_candidates(self, candidate_indices: torch.Tensor) -> torch.Tensor:
        batch = self._last_reassign_batch
        assert batch is not None
        ts = batch["ts"].long()
        w2cs = batch["w2cs"]
        Ks = batch["Ks"]
        masks = batch["masks"]
        dilated_masks = batch["dilated_masks"]
        depths = batch["depths"]

        points = self.model.fg.params["means"].detach()[candidate_indices]
        M = int(points.shape[0])
        K = int(self.model.num_motion_bases)
        B = int(ts.shape[0])
        dtype = points.dtype
        device = points.device
        H, W = masks.shape[-2:]

        base_rots, base_transls = self.model.motion_bases.compute_base_transforms(ts)
        posed = torch.einsum("kbij,mj->mkbi", base_rots, points) + base_transls[None]

        visible = torch.zeros(M, K, device=device, dtype=dtype)
        outside = torch.zeros_like(visible)
        depth_sum = torch.zeros_like(visible)
        depth_count = torch.zeros_like(visible)

        for b in range(B):
            pts = posed[:, :, b].reshape(M * K, 3)
            pts_h = F.pad(pts, (0, 1), value=1.0)
            cam = torch.einsum("ij,nj->ni", w2cs[b, :3], pts_h)
            z = cam[:, 2]
            valid_z = z > 1e-6
            x = Ks[b, 0, 0] * (cam[:, 0] / z.clamp_min(1e-6)) + Ks[b, 0, 2]
            y = Ks[b, 1, 1] * (cam[:, 1] / z.clamp_min(1e-6)) + Ks[b, 1, 2]
            xi = torch.round(x).long()
            yi = torch.round(y).long()
            in_img = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H) & valid_z
            in_img_mk = in_img.reshape(M, K)
            visible += in_img_mk.to(dtype)
            if not in_img.any():
                continue

            flat_idx = torch.where(in_img)[0]
            local_m = flat_idx // K
            local_k = flat_idx % K
            px = xi[flat_idx]
            py = yi[flat_idx]
            dilated_at = dilated_masks[b, py, px] > 0.5
            fg_at = masks[b, py, px] > 0.5
            outside[local_m, local_k] += (~dilated_at).to(dtype)

            if fg_at.any():
                depth_idx = flat_idx[fg_at]
                depth_m = local_m[fg_at]
                depth_k = local_k[fg_at]
                gt_depth = depths[b, py[fg_at], px[fg_at]].to(dtype)
                z_sel = z[depth_idx]
                depth_abs = (z_sel - gt_depth).abs()
                tol = torch.maximum(
                    torch.full_like(gt_depth, float(self.ghost_depth_abs_thresh)),
                    gt_depth.abs() * float(self.ghost_depth_rel_thresh),
                ).clamp_min(1e-4)
                depth_err = (depth_abs / tol).clamp(max=5.0)
                depth_sum[depth_m, depth_k] += depth_err
                depth_count[depth_m, depth_k] += 1.0

        visible_clamped = visible.clamp_min(1.0)
        outside_ratio = outside / visible_clamped
        depth_cost = depth_sum / depth_count.clamp_min(1.0)
        invalid_ratio = 1.0 - visible / float(max(B, 1))
        knn_cost = self._assignment_neighbor_cost(candidate_indices)

        current = self.model.fg.params["motion_coefs"].detach().argmax(dim=-1)[
            candidate_indices
        ]
        switch_cost = torch.ones(M, K, device=device, dtype=dtype) * float(
            self.reassign_switch_penalty
        )
        switch_cost[torch.arange(M, device=device), current] = 0.0

        return (
            self.w_reassign_mask * outside_ratio
            + self.w_reassign_depth * depth_cost
            + self.w_reassign_knn * knn_cost
            + 0.25 * invalid_ratio
            + switch_cost
        )

    @torch.no_grad()
    def _assignment_neighbor_cost(self, candidate_indices: torch.Tensor) -> torch.Tensor:
        logits = self.model.fg.params["motion_coefs"].detach()
        M = int(candidate_indices.shape[0])
        K = int(logits.shape[1])
        cost = logits.new_zeros((M, K))
        edges = self.model.assignment_graph_edges
        if edges.numel() == 0 or M == 0:
            return cost
        device = logits.device
        edges = edges.to(device)
        weights = self.model.assignment_graph_weights.to(device=device, dtype=logits.dtype)
        if weights.numel() != edges.shape[0]:
            weights = torch.ones(edges.shape[0], device=device, dtype=logits.dtype)
        candidate_pos = torch.full(
            (self.model.num_fg_gaussians,),
            -1,
            device=device,
            dtype=torch.long,
        )
        candidate_pos[candidate_indices.to(device)] = torch.arange(M, device=device)
        labels = logits.argmax(dim=-1)
        denom = logits.new_zeros(M)

        def _accumulate(endpoint: int, neighbor_endpoint: int):
            pos = candidate_pos[edges[:, endpoint]]
            valid = pos >= 0
            if not valid.any():
                return
            pos_valid = pos[valid]
            neigh_labels = labels[edges[valid, neighbor_endpoint]]
            w = weights[valid]
            disagree = 1.0 - F.one_hot(neigh_labels, num_classes=K).to(logits.dtype)
            cost.index_add_(0, pos_valid, disagree * w[:, None])
            denom.index_add_(0, pos_valid, w)

        _accumulate(0, 1)
        _accumulate(1, 0)
        return cost / denom[:, None].clamp_min(1e-8)

    def _write_reassignment_report(
        self,
        candidate_indices: torch.Tensor,
        switched_from: torch.Tensor,
        switched_to: torch.Tensor,
        cost_report: dict,
        prior_report: dict,
        protected_count: int,
    ):
        K = self.model.num_motion_bases
        switch_matrix = torch.zeros(K, K, dtype=torch.long)
        if switched_from.numel() > 0:
            for src, dst in zip(switched_from.cpu().tolist(), switched_to.cpu().tolist()):
                switch_matrix[int(src), int(dst)] += 1
        labels = self.model.fg.params["motion_coefs"].detach().argmax(dim=-1)
        label_counts = torch.bincount(labels.cpu(), minlength=K)
        report = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "candidate_count": int(candidate_indices.numel()),
            "switched_count": int(switched_to.numel()),
            "protected_high_opacity_ghost_count": int(protected_count),
            "label_counts": label_counts.tolist(),
            "switch_matrix_from_to": switch_matrix.tolist(),
            "cost": cost_report,
            "part_prior": prior_report,
            "config": {
                "reassign_candidate_score_thresh": float(
                    self.reassign_candidate_score_thresh
                ),
                "reassign_min_opacity": float(self.reassign_min_opacity),
                "reassign_improve_ratio": float(self.reassign_improve_ratio),
                "ghost_cull_opacity_thresh": float(self.ghost_cull_opacity_thresh),
            },
        }
        out_path = Path(self.work_dir) / "stage2_reassignment_report.yaml"
        with open(out_path, "w") as f:
            yaml.safe_dump(report, f, default_flow_style=False)

    @torch.no_grad()
    def _run_stage2_ghost_cull(self):
        if (
            not self.enable_stage2_ghost_cull
            or not self.enable_ghost_score
            or self.epoch < self.ghost_start_epoch
            or self.ghost_cull_every_steps <= 0
            or self.global_step == 0
            or self.global_step % self.ghost_cull_every_steps != 0
        ):
            return

        score = self.model.ghost_score
        visible = self.model.ghost_visible_count
        opacity = self.model.fg.get_opacities().reshape(self.model.num_fg_gaussians, -1).mean(dim=-1)
        attempted = self.model.reassign_attempt_count > 0
        if not self.ghost_cull_requires_reassign_attempt:
            attempted = torch.ones_like(attempted, dtype=torch.bool)
        protected = (
            (score >= self.ghost_cull_score_thresh)
            & (opacity >= self.ghost_cull_opacity_thresh)
            & (visible >= self.ghost_min_visible)
        )
        should_cull = (
            (score >= self.ghost_cull_score_thresh)
            & (opacity < self.ghost_cull_opacity_thresh)
            & (visible >= self.ghost_min_visible)
            & attempted.to(score.device)
        )
        self._last_reassignment_stats["ghost_protected_count"] = float(
            protected.sum().item()
        )
        if not should_cull.any():
            return

        if should_cull.sum().item() > self.max_ghost_cull_per_step:
            cull_indices = torch.where(should_cull)[0]
            top_local = torch.topk(
                score[cull_indices],
                k=int(self.max_ghost_cull_per_step),
                largest=True,
            ).indices
            limited = torch.zeros_like(should_cull)
            limited[cull_indices[top_local]] = True
            should_cull = limited

        old_num_fg = self.model.num_fg_gaussians
        old_num_total = self.model.num_gaussians
        full_keep = torch.ones(old_num_total, device=should_cull.device, dtype=torch.bool)
        full_keep[:old_num_fg] = ~should_cull

        fg_param_map = self.model.cull_foreground_gaussians(should_cull)
        for param_name, new_param in fg_param_map.items():
            opt_name = f"fg.params.{param_name}"
            optimizer = self.optimizers.get(opt_name)
            if optimizer is not None:
                self._replace_param_in_optimizer_after_cull(optimizer, new_param, should_cull)

        for key, value in list(self.running_stats.items()):
            if value.shape[0] == full_keep.shape[0]:
                self.running_stats[key] = value[full_keep.to(value.device)]

        self._last_reassignment_stats["ghost_culled_count"] = float(
            should_cull.sum().item()
        )
        self.writer.add_scalar(
            "train/ghost_culled_count",
            float(should_cull.sum().item()),
            self.global_step,
        )
        guru.info(
            "Stage2 ghost cull removed "
            f"{int(should_cull.sum().item())} foreground gaussians; "
            f"{self.model.num_fg_gaussians} foreground gaussians left"
        )

    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().astype(np.float32)

    def _setup_joint_axis_viewer(self):
        if self.viewer is None:
            return
        server = self.viewer.server
        init = self.model.get_joint_viz_segments(use_initial=True)
        current = self.model.get_joint_viz_segments(use_initial=False)
        self._add_joint_axis_nodes(
            server,
            prefix="/articulat3d_stage2/joints/init",
            data=init,
            red=(140, 0, 0),
            green=(0, 140, 0),
            line_width=self.joint_axis_init_line_width,
        )
        self._add_joint_axis_nodes(
            server,
            prefix="/articulat3d_stage2/joints/current",
            data=current,
            red=(255, 0, 0),
            green=(0, 255, 0),
            line_width=self.joint_axis_current_line_width,
        )
        if self.show_gt_joint_axes:
            gt = self.model.get_gt_joint_viz_segments()
            self._add_gt_joint_axis_nodes(
                server,
                prefix="/articulat3d_stage2/joints/gt",
                data=gt,
                blue=(0, 80, 255),
                line_width=self.gt_joint_axis_line_width,
            )
        guru.info("Added Articulat3D stage2 joint axes to viser scene")

    def _part_prior_center_data(self) -> tuple[np.ndarray, np.ndarray]:
        self.model.update_part_priors(
            conf_thresh=self.part_prior_conf_thresh,
            opacity_thresh=self.part_prior_opacity_thresh,
            ghost_score_thresh=self.part_prior_ghost_score_thresh,
            min_points=self.part_prior_min_points,
            scale_min=self.part_prior_scale_min,
            center_mode=self.part_prior_center_mode,
            obb_low_quantile=self.part_prior_obb_low_quantile,
            obb_high_quantile=self.part_prior_obb_high_quantile,
        )
        ts = self._viewer_current_timestep(default=0)
        centers, valid = self.model.get_part_prior_centers(
            ts=ts,
            posed=self.viewer_part_centers_follow_motion,
        )
        colors = self._part_color_palette(self.model.num_motion_bases)[valid.cpu().numpy()]
        return self._to_numpy(centers), colors

    @staticmethod
    def _part_color_palette(num_parts: int) -> np.ndarray:
        base = np.asarray(
            [
                (220, 220, 220),
                (255, 64, 64),
                (255, 176, 0),
                (0, 220, 80),
                (64, 128, 255),
                (220, 64, 255),
                (0, 220, 220),
                (160, 96, 32),
                (255, 128, 192),
                (128, 255, 64),
            ],
            dtype=np.uint8,
        )
        if num_parts <= len(base):
            return base[:num_parts]
        reps = int(np.ceil(num_parts / len(base)))
        return np.tile(base, (reps, 1))[:num_parts]

    def _setup_part_prior_center_viewer(self):
        if self.viewer is None:
            return
        points, colors = self._part_prior_center_data()
        if len(points) == 0:
            return
        handle = self.viewer.server.scene.add_point_cloud(
            "/articulat3d_stage2/part_prior/centers",
            points,
            colors,
            point_size=self.part_prior_center_point_size,
            point_shape="circle",
        )
        self._joint_axis_handles["/articulat3d_stage2/part_prior/centers"] = handle
        self._register_part_prior_center_timestep_callback()
        guru.info("Added Articulat3D stage2 part prior centers to viser scene")

    def _register_part_prior_center_timestep_callback(self):
        if self._part_prior_center_timestep_callback_registered or self.viewer is None:
            return
        try:
            playback_guis = getattr(self.viewer, "_playback_guis", None)
            if not playback_guis:
                return

            @playback_guis[0].on_update
            def _(_event):
                self._update_part_prior_center_viewer()

            canonical = getattr(self.viewer, "_canonical_checkbox", None)
            if canonical is not None:
                canonical.on_update(lambda _event: self._update_part_prior_center_viewer())
            self._part_prior_center_timestep_callback_registered = True
        except Exception as exc:
            guru.warning(f"Could not attach part prior center viewer callbacks: {exc}")

    def _add_joint_axis_nodes(
        self,
        server,
        prefix: str,
        data: dict[str, torch.Tensor],
        red: tuple[int, int, int],
        green: tuple[int, int, int],
        line_width: float,
    ):
        prismatic = self._to_numpy(data["prismatic_segments"])
        revolute = self._to_numpy(data["revolute_segments"])
        pivots = self._to_numpy(data["revolute_pivots"])
        if len(prismatic) > 0:
            self._joint_axis_handles[f"{prefix}/prismatic_axes"] = server.scene.add_line_segments(
                f"{prefix}/prismatic_axes",
                prismatic,
                red,
                line_width=line_width,
            )
        if len(revolute) > 0:
            self._joint_axis_handles[f"{prefix}/revolute_axes"] = server.scene.add_line_segments(
                f"{prefix}/revolute_axes",
                revolute,
                green,
                line_width=line_width,
            )
        if len(pivots) > 0:
            self._joint_axis_handles[f"{prefix}/revolute_pivots"] = server.scene.add_point_cloud(
                f"{prefix}/revolute_pivots",
                pivots,
                green,
                point_size=self.joint_pivot_point_size,
                point_shape="circle",
            )

    def _add_gt_joint_axis_nodes(
        self,
        server,
        prefix: str,
        data: dict[str, torch.Tensor],
        blue: tuple[int, int, int],
        line_width: float,
    ):
        prismatic = self._to_numpy(data["prismatic_segments"])
        revolute = self._to_numpy(data["revolute_segments"])
        pivots = self._to_numpy(data["revolute_pivots"])
        if len(prismatic) > 0:
            self._joint_axis_handles[f"{prefix}/prismatic_axes"] = server.scene.add_line_segments(
                f"{prefix}/prismatic_axes",
                prismatic,
                blue,
                line_width=line_width,
            )
        if len(revolute) > 0:
            self._joint_axis_handles[f"{prefix}/revolute_axes"] = server.scene.add_line_segments(
                f"{prefix}/revolute_axes",
                revolute,
                blue,
                line_width=line_width,
            )
        if len(pivots) > 0:
            self._joint_axis_handles[f"{prefix}/revolute_pivots"] = server.scene.add_point_cloud(
                f"{prefix}/revolute_pivots",
                pivots,
                blue,
                point_size=self.joint_pivot_point_size,
                point_shape="circle",
            )

    def _update_current_joint_axis_viewer(self):
        if self.viewer is None:
            return
        data = self.model.get_joint_viz_segments(use_initial=False)
        mapping = {
            "/articulat3d_stage2/joints/current/prismatic_axes": data["prismatic_segments"],
            "/articulat3d_stage2/joints/current/revolute_axes": data["revolute_segments"],
            "/articulat3d_stage2/joints/current/revolute_pivots": data["revolute_pivots"],
        }
        for name, tensor in mapping.items():
            handle = self._joint_axis_handles.get(name)
            if handle is None:
                continue
            handle.points = self._to_numpy(tensor)

    def _update_part_prior_center_viewer(self):
        if self.viewer is None:
            return
        points, colors = self._part_prior_center_data()
        handle = self._joint_axis_handles.get("/articulat3d_stage2/part_prior/centers")
        if handle is None:
            if len(points) == 0:
                return
            self._setup_part_prior_center_viewer()
            return
        handle.points = points
        if hasattr(handle, "colors"):
            handle.colors = colors
