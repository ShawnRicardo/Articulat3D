import time
from typing import cast

import torch
import torch.nn.functional as F
from loguru import logger as guru

import flow3d.trainer_3d as trainer_3d_module
from flow3d.configs import LossesConfig, OptimizerConfig, SceneLRConfig
from flow3d.init_utils_articulat3d import compute_segmented_se3_smoothness_loss
from flow3d.loss_utils import (
    compute_gradient_loss,
    compute_z_acc_loss,
    masked_l1_loss,
)
from flow3d.scene_model import SceneModel
from flow3d.trainer_3d import Trainer


class _NoOpLPIPS(torch.nn.Module):
    def reset(self):
        pass

    def update(self, *args, **kwargs):
        pass

    def compute(self):
        return 0.0


def disable_lpips_metric_downloads():
    trainer_3d_module.mLPIPS = _NoOpLPIPS
    try:
        import flow3d.validator as validator_module

        validator_module.mLPIPS = _NoOpLPIPS
    except Exception:
        pass


class TrainerArticulat3D(Trainer):
    def __init__(
        self,
        model: SceneModel,
        device: torch.device,
        lr_cfg: SceneLRConfig,
        losses_cfg: LossesConfig,
        optim_cfg: OptimizerConfig,
        work_dir: str,
        port: int | None = None,
        motion_segment_length: int = 200,
        **kwargs,
    ):
        self.motion_segment_length = motion_segment_length
        disable_lpips_metric_downloads()
        super().__init__(
            model=model,
            device=device,
            lr_cfg=lr_cfg,
            losses_cfg=losses_cfg,
            optim_cfg=optim_cfg,
            work_dir=work_dir,
            port=port,
            **kwargs,
        )

    @staticmethod
    def init_from_checkpoint(
        path: str, device: torch.device, use_2dgs, *args, **kwargs
    ) -> tuple["TrainerArticulat3D", int]:
        guru.info(f"Loading checkpoint from {path}")
        ckpt = torch.load(path, weights_only=False)
        state_dict = ckpt["model"]
        model = SceneModel.init_from_state_dict(state_dict)
        model = model.to(device)
        model.use_2dgs = use_2dgs
        trainer = TrainerArticulat3D(model, device, *args, **kwargs)
        if "optimizers" in ckpt:
            trainer.load_checkpoint_optimizers(ckpt["optimizers"])
        if "schedulers" in ckpt:
            trainer.load_checkpoint_schedulers(ckpt["schedulers"])
        trainer.global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0)
        trainer.set_epoch(start_epoch)
        return trainer, start_epoch

    def _interior_ts_mask(self, ts: torch.Tensor, num_frames: int) -> torch.Tensor:
        mask = (ts > 0) & (ts < num_frames - 1)
        if self.motion_segment_length > 0:
            phase = ts % self.motion_segment_length
            mask = mask & (phase > 0) & (phase < self.motion_segment_length - 1)
        return mask

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
            cos_sim = torch.sum(rendered_normals * surf_normals, dim=-1)
            loss += (1 - cos_sim).mean() * 0.05

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
            ) + masked_l1_loss(
                rendered_all["mask"],
                masks[..., None],
                quantile=0.98,
            )
        loss += mask_loss * self.losses_cfg.w_mask

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

        small_accel_loss = compute_segmented_se3_smoothness_loss(
            self.model.motion_bases.params["rots"],
            self.model.motion_bases.params["transls"],
            self.motion_segment_length,
        )
        loss += small_accel_loss * self.losses_cfg.w_smooth_bases

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

        if self.model.fg.params["means"].isnan().sum() > 0:
            import ipdb

            ipdb.set_trace()

        stats = {
            "train/loss": loss.item(),
            "train/rgb_loss": rgb_loss.item(),
            "train/mask_loss": mask_loss.item(),
            "train/depth_loss": depth_loss.item(),
            "train/depth_gradient_loss": depth_gradient_loss.item(),
            "train/small_accel_loss": small_accel_loss.item(),
            "train/small_accel_loss_tracks": small_accel_loss_tracks.item(),
            "train/z_acc_loss": z_accel_loss.item(),
            "train/num_gaussians": self.model.num_gaussians,
            "train/num_fg_gaussians": self.model.num_fg_gaussians,
            "train/num_bg_gaussians": self.model.num_bg_gaussians,
        }

        with torch.no_grad():
            psnr = self.psnr_metric(
                rendered_imgs, imgs, masks if not self.model.has_bg else valid_masks
            )
            self.psnr_metric.reset()
            stats["train/psnr"] = psnr
            if self.model.has_bg:
                bg_psnr = self.bg_psnr_metric(rendered_imgs, imgs, 1.0 - masks)
                fg_psnr = self.fg_psnr_metric(rendered_imgs, imgs, masks)
                self.bg_psnr_metric.reset()
                self.fg_psnr_metric.reset()
                stats["train/bg_psnr"] = bg_psnr
                stats["train/fg_psnr"] = fg_psnr

        stats.update(
            **{
                "train/num_rays_per_sec": num_rays_per_sec,
                "train/num_rays_per_step": float(num_rays_per_step),
            }
        )
        return loss, stats, num_rays_per_step, num_rays_per_sec
