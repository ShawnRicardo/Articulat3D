import functools
import os
import os.path as osp
import time
from dataclasses import asdict
from typing import cast

import imageio as iio
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as guru
from nerfview import CameraState, Viewer
from pytorch_msssim import SSIM
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from flow3d.configs import LossesConfig, OptimizerConfig, SceneLRConfig
from flow3d.data.utils import normalize_coords, to_device
from flow3d.metrics import PCK, compute_psnr, mLPIPS, mPSNR, mSSIM
from flow3d.scene_model import SceneModel
from flow3d.vis.utils import (
    apply_depth_colormap,
    make_video_divisble,
    plot_correspondences,
)


class Validator:
    def __init__(
        self,
        model: SceneModel,
        device: torch.device,
        train_loader: DataLoader | None,
        val_img_loader: DataLoader | None,
        val_kpt_loader: DataLoader | None,
        save_dir: str,
        depth_alpha_vis_thresh: float = 0.15,
    ):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_img_loader = val_img_loader
        self.val_kpt_loader = val_kpt_loader
        self.save_dir = save_dir
        self.has_bg = self.model.has_bg
        self.depth_alpha_vis_thresh = depth_alpha_vis_thresh

        # metrics
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.psnr_metric = mPSNR()
        self.ssim_metric = mSSIM()
        self.lpips_metric = mLPIPS().to(device)
        self.fg_psnr_metric = mPSNR()
        self.fg_ssim_metric = mSSIM()
        self.fg_lpips_metric = mLPIPS().to(device)
        self.bg_psnr_metric = mPSNR()
        self.bg_ssim_metric = mSSIM()
        self.bg_lpips_metric = mLPIPS().to(device)
        self.pck_metric = PCK()

    def reset_metrics(self):
        self.psnr_metric.reset()
        self.ssim_metric.reset()
        self.lpips_metric.reset()
        self.fg_psnr_metric.reset()
        self.fg_ssim_metric.reset()
        self.fg_lpips_metric.reset()
        self.bg_psnr_metric.reset()
        self.bg_ssim_metric.reset()
        self.bg_lpips_metric.reset()
        self.pck_metric.reset()

    def _init_video_lpips_metric(self):
        try:
            # Articulat3D training patches validator.mLPIPS to avoid an eager
            # AlexNet download at startup. For saved-video metrics we explicitly
            # use the real metric and allow the one-time LPIPS weight load here.
            from flow3d.metrics import mLPIPS as RealLPIPS

            return RealLPIPS().to(self.device)
        except Exception as exc:
            guru.warning(f"LPIPS metric unavailable for train-video dump: {exc}")
            return None

    def _compute_image_metrics_for_video(
        self,
        rendered_img: torch.Tensor,
        target_img: torch.Tensor,
        metric_mask: torch.Tensor,
        ssim_metric: mSSIM,
        lpips_metric,
    ) -> dict[str, float]:
        rendered_img = rendered_img.clamp(0.0, 1.0)
        target_img = target_img.clamp(0.0, 1.0)
        metric_mask = metric_mask.float()

        psnr = compute_psnr(rendered_img, target_img, metric_mask)

        ssim_metric.reset()
        ssim_metric.update(rendered_img, target_img, metric_mask)
        ssim = float(ssim_metric.compute().detach().cpu())

        lpips = float("nan")
        if lpips_metric is not None:
            try:
                scores = lpips_metric.net(
                    (rendered_img * metric_mask[..., None]).permute(0, 3, 1, 2),
                    (target_img * metric_mask[..., None]).permute(0, 3, 1, 2),
                    normalize=True,
                )
                lpips = float(
                    ((scores * metric_mask[:, None]).sum() / metric_mask.sum().clamp(min=1.0))
                    .detach()
                    .cpu()
                )
            except Exception as exc:
                guru.warning(f"Failed to compute LPIPS for a train-video frame: {exc}")
        return {"psnr": psnr, "ssim": ssim, "lpips": lpips}

    def _write_video_metrics(self, video_dir: str, metric_rows: list[dict[str, float | str]]):
        if not metric_rows:
            return
        metrics_path = osp.join(video_dir, "metrics.txt")
        numeric_keys = ["psnr", "ssim", "lpips"]
        with open(metrics_path, "w") as f:
            f.write("frame\tpsnr\tssim\tlpips\n")
            for row in metric_rows:
                f.write(
                    f"{row['frame']}\t"
                    f"{float(row['psnr']):.6f}\t"
                    f"{float(row['ssim']):.6f}\t"
                    f"{float(row['lpips']):.6f}\n"
                )
            f.write("\n")
            f.write("mean")
            for key in numeric_keys:
                vals = np.array([float(row[key]) for row in metric_rows], dtype=np.float64)
                f.write(f"\t{np.nanmean(vals):.6f}")
            f.write("\n")

    def _get_motion_coefs_for_video(self) -> torch.Tensor:
        if hasattr(self.model, "get_motion_coefs"):
            try:
                return self.model.get_motion_coefs(deterministic=True)
            except TypeError:
                return self.model.get_motion_coefs()
        return self.model.fg.get_coefs()

    @staticmethod
    def _score_heatmap_colors(score: torch.Tensor) -> torch.Tensor:
        score = score.detach().clamp(0.0, 1.0)
        heat = score.sqrt()
        low_score_color = torch.tensor(
            [0.01, 0.02, 0.10],
            dtype=score.dtype,
            device=score.device,
        )
        high_score_color = torch.tensor(
            [1.00, 0.22, 0.00],
            dtype=score.dtype,
            device=score.device,
        )
        return (
            low_score_color[None] * (1.0 - heat[:, None])
            + high_score_color[None] * heat[:, None]
        )

    def _get_assignment_confidence_colors_for_video(self) -> torch.Tensor:
        if "motion_coefs" in self.model.fg.params:
            logits = self.model.fg.params["motion_coefs"].detach()
            probs = F.softmax(logits, dim=-1)
        else:
            probs = self._get_motion_coefs_for_video().detach()
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        confidence = probs.max(dim=-1).values.clamp(0.0, 1.0)
        return self._score_heatmap_colors(1.0 - confidence)

    def _get_ghost_score_colors_for_video(self) -> torch.Tensor | None:
        ghost_score = getattr(self.model, "ghost_score", None)
        if ghost_score is None:
            return None
        ghost_score = ghost_score.detach()
        if ghost_score.ndim > 1:
            ghost_score = ghost_score.reshape(ghost_score.shape[0], -1).mean(dim=-1)
        num_fg = getattr(self.model, "num_fg_gaussians", self.model.fg.num_gaussians)
        if ghost_score.shape[0] != num_fg:
            guru.warning(
                "Skipping ghost score video because ghost_score shape "
                f"{tuple(ghost_score.shape)} does not match num_fg_gaussians={num_fg}"
            )
            return None
        return self._score_heatmap_colors(ghost_score)

    @torch.no_grad()
    def validate(self):
        # 清 0 累计的评价指标
        self.reset_metrics()
        # 渲染并评估图像相关的指标/可视化
        metric_imgs = self.validate_imgs() or {}
        # 渲染关键点 / 轨迹类的指标，不重要，我们其实并不需要他的轨迹正确
        metric_kpts = self.validate_keypoints() or {}
        return {**metric_imgs, **metric_kpts}

    @torch.no_grad()
    def validate_imgs(self):
        guru.info("rendering validation images...")
        if self.val_img_loader is None:
            return
        per_img_psnrs = []
        for batch in tqdm(self.val_img_loader, desc="render val images"):
            batch = to_device(batch, self.device)
            frame_name = batch["frame_names"][0]
            t = batch["ts"][0]
            # (1, 4, 4).
            w2c = batch["w2cs"]
            # (1, 3, 3).
            K = batch["Ks"]
            # (1, H, W, 3).
            img = batch["imgs"]
            # (1, H, W).
            valid_mask = batch.get(
                "valid_masks", torch.ones_like(batch["imgs"][..., 0])
            )
            # (1, H, W).
            fg_mask = batch["masks"]

            # (H, W).
            covisible_mask = batch.get(
                "covisible_masks",
                torch.ones_like(fg_mask)[None],
            )
            W, H = img_wh = img[0].shape[-2::-1]
            rendered = self.model.render(t, w2c, K, img_wh, return_depth=True)
            
            # Compute metrics.
            valid_mask *= covisible_mask
            fg_valid_mask = fg_mask * valid_mask
            bg_valid_mask = (1 - fg_mask) * valid_mask
            main_valid_mask = valid_mask if self.has_bg else fg_valid_mask

            self.psnr_metric.update(rendered["img"], img, main_valid_mask)
            self.ssim_metric.update(rendered["img"], img, main_valid_mask)
            self.lpips_metric.update(rendered["img"], img, main_valid_mask)

            if self.has_bg:
                self.fg_psnr_metric.update(rendered["img"], img, fg_valid_mask)
                self.fg_ssim_metric.update(rendered["img"], img, fg_valid_mask)
                self.fg_lpips_metric.update(rendered["img"], img, fg_valid_mask)

                self.bg_psnr_metric.update(rendered["img"], img, bg_valid_mask)
                self.bg_ssim_metric.update(rendered["img"], img, bg_valid_mask)
                self.bg_lpips_metric.update(rendered["img"], img, bg_valid_mask)

            # Dump results.
            results_dir = osp.join(self.save_dir, "results", "rgb")
            os.makedirs(results_dir, exist_ok=True)
            iio.imwrite(
                osp.join(results_dir, f"{frame_name}.png"),
                (rendered["img"][0].cpu().numpy() * 255).astype(np.uint8),
            )

        return {
            "val/psnr": self.psnr_metric.compute(),
            "val/ssim": self.ssim_metric.compute(),
            "val/lpips": self.lpips_metric.compute(),
            "val/fg_psnr": self.fg_psnr_metric.compute(),
            "val/fg_ssim": self.fg_ssim_metric.compute(),
            "val/fg_lpips": self.fg_lpips_metric.compute(),
            "val/bg_psnr": self.bg_psnr_metric.compute(),
            "val/bg_ssim": self.bg_ssim_metric.compute(),
            "val/bg_lpips": self.bg_lpips_metric.compute(),
        }

    @torch.no_grad()
    def validate_keypoints(self):
        if self.val_kpt_loader is None:
            return
        pred_keypoints_3d_all = []
        time_ids = self.val_kpt_loader.dataset.time_ids.tolist()
        h, w = self.val_kpt_loader.dataset.dataset.imgs.shape[1:3]
        pred_train_depths = np.zeros((len(time_ids), h, w))

        for batch in tqdm(self.val_kpt_loader, desc="render val keypoints"):
            batch = to_device(batch, self.device)
            # (2,).
            ts = batch["ts"][0]
            # (2, 4, 4).
            w2cs = batch["w2cs"][0]
            # (2, 3, 3).
            Ks = batch["Ks"][0]
            # (2, H, W, 3).
            imgs = batch["imgs"][0]
            # (2, P, 3).
            keypoints = batch["keypoints"][0]
            # (P,)
            keypoint_masks = (keypoints[..., -1] > 0.5).all(dim=0)
            src_keypoints, target_keypoints = keypoints[:, keypoint_masks, :2]
            W, H = img_wh = imgs.shape[-2:0:-1]
            rendered = self.model.render(
                ts[0].item(),
                w2cs[:1],
                Ks[:1],
                img_wh,
                target_ts=ts[1:],
                target_w2cs=w2cs[1:],
                return_depth=True,
            )
            pred_tracks_3d = rendered["tracks_3d"][0, ..., 0, :]
            pred_tracks_2d = torch.einsum("ij,hwj->hwi", Ks[1], pred_tracks_3d)
            pred_tracks_2d = pred_tracks_2d[..., :2] / torch.clamp(
                pred_tracks_2d[..., -1:], min=1e-6
            )
            pred_keypoints = F.grid_sample(
                pred_tracks_2d[None].permute(0, 3, 1, 2),
                normalize_coords(src_keypoints, H, W)[None, None],
                align_corners=True,
            ).permute(0, 2, 3, 1)[0, 0]

            # Compute metrics.
            self.pck_metric.update(pred_keypoints, target_keypoints, max(img_wh) * 0.05)

            padded_keypoints_3d = torch.zeros_like(keypoints[0])
            pred_keypoints_3d = F.grid_sample(
                pred_tracks_3d[None].permute(0, 3, 1, 2),
                normalize_coords(src_keypoints, H, W)[None, None],
                align_corners=True,
            ).permute(0, 2, 3, 1)[0, 0]
            # Transform 3D keypoints back to world space.
            pred_keypoints_3d = torch.einsum(
                "ij,pj->pi",
                torch.linalg.inv(w2cs[1])[:3],
                F.pad(pred_keypoints_3d, (0, 1), value=1.0),
            )
            padded_keypoints_3d[keypoint_masks] = pred_keypoints_3d
            # Cache predicted keypoints.
            pred_keypoints_3d_all.append(padded_keypoints_3d.cpu().numpy())
            pred_train_depths[time_ids.index(ts[0].item())] = (
                rendered["depth"][0, ..., 0].cpu().numpy()
            )

        # Dump unified results.
        all_Ks = self.val_kpt_loader.dataset.dataset.Ks
        all_w2cs = self.val_kpt_loader.dataset.dataset.w2cs

        keypoint_result_dict = {
            "Ks": all_Ks[time_ids].cpu().numpy(),
            "w2cs": all_w2cs[time_ids].cpu().numpy(),
            "pred_keypoints_3d": np.stack(pred_keypoints_3d_all, 0),
            "pred_train_depths": pred_train_depths,
        }

        results_dir = osp.join(self.save_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        np.savez(
            osp.join(results_dir, "keypoints.npz"),
            **keypoint_result_dict,
        )
        guru.info(
            f"Dumped keypoint results to {results_dir=} {keypoint_result_dict['pred_keypoints_3d'].shape=}"
        )

        return {"val/pck": self.pck_metric.compute()}

    @torch.no_grad()
    def save_train_videos(self, epoch: int):
        if self.train_loader is None:
            return
        video_dir = osp.join(self.save_dir, "videos", f"epoch_{epoch:04d}")
        os.makedirs(video_dir, exist_ok=True)
        images_dir = osp.join(video_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        fps = getattr(self.train_loader.dataset.dataset, "fps", 15.0)
        video_ssim_metric = mSSIM().to(self.device)
        video_lpips_metric = self._init_video_lpips_metric()
        metric_rows: list[dict[str, float | str]] = []
        # Render video.
        video = []
        ref_pred_depths = []
        ref_pred_depths_alpha_masked = []
        masks = []
        depth_min, depth_max = 1e6, 0
        normals = []
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Rendering video", leave=False)):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # [1] 这个 t 其实就是一个时刻
            t = batch["ts"][0]
            # (4, 4).
            w2c = batch["w2cs"][0]
            # (3, 3).
            K = batch["Ks"][0]
            # (H, W, 3).
            img = batch["imgs"][0]
            # (H, W).
            depth = batch["depths"][0]
            valid_mask = batch.get(
                "valid_masks",
                torch.ones_like(batch["imgs"][..., 0]),
            )[0]
            # (H, W).
            img_wh = img.shape[-2::-1]
            rendered = self.model.render(
                t, w2c[None], K[None], img_wh, return_depth=True, return_mask=True
            )
            rendered_img = rendered["img"][0]

            frame_name = batch["frame_names"][0]
            if isinstance(frame_name, (list, tuple)):
                frame_name = frame_name[0]
            frame_name = str(frame_name)
            img_np = (img.clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
            rendered_np = (
                rendered_img.clamp(0.0, 1.0).cpu().numpy() * 255
            ).astype(np.uint8)
            pair_np = np.concatenate([img_np, rendered_np], axis=1)
            iio.imwrite(osp.join(images_dir, f"{frame_name}.png"), pair_np)

            frame_metrics = self._compute_image_metrics_for_video(
                rendered_img[None],
                img[None],
                valid_mask[None],
                video_ssim_metric,
                video_lpips_metric,
            )
            metric_rows.append({"frame": frame_name, **frame_metrics})

            # Putting results onto CPU since it will consume unnecessarily
            # large GPU memory for long sequence OW.
            video.append(torch.cat([img, rendered_img], dim=1).cpu())
            ref_pred_depth = torch.cat(
                (depth[..., None], rendered["depth"][0]), dim=1
            ).cpu()
            ref_pred_depths.append(ref_pred_depth)
            rendered_depth_alpha_masked = rendered["depth"][0].clone()
            rendered_acc = rendered["acc"][0]
            rendered_depth_alpha_masked = torch.where(
                rendered_acc >= self.depth_alpha_vis_thresh,
                rendered_depth_alpha_masked,
                depth[..., None],
            )
            ref_pred_depth_alpha_masked = torch.cat(
                (depth[..., None], rendered_depth_alpha_masked), dim=1
            ).cpu()
            ref_pred_depths_alpha_masked.append(ref_pred_depth_alpha_masked)
            depth_min = min(depth_min, ref_pred_depth.min().item())
            depth_max = max(depth_max, ref_pred_depth.quantile(0.99).item())
            if rendered["mask"] is not None:
                masks.append(rendered["mask"][0].cpu().squeeze(-1))
            if rendered["rend_normal"] is not None:
                normals.append(rendered["rend_normal"].cpu())

        self._write_video_metrics(video_dir, metric_rows)

        # rgb video
        video = torch.stack(video, dim=0)
        iio.mimwrite(
            osp.join(video_dir, "rgbs.mp4"),
            make_video_divisble((video.numpy() * 255).astype(np.uint8)),
            fps=fps,
        )
        # depth video
        depth_video = torch.stack(
            [
                apply_depth_colormap(
                    ref_pred_depth, near_plane=depth_min, far_plane=depth_max
                )
                for ref_pred_depth in ref_pred_depths
            ],
            dim=0,
        )
        iio.mimwrite(
            osp.join(video_dir, "depths.mp4"),
            make_video_divisble((depth_video.numpy() * 255).astype(np.uint8)),
            fps=fps,
        )
        depth_alpha_masked_video = torch.stack(
            [
                apply_depth_colormap(
                    ref_pred_depth, near_plane=depth_min, far_plane=depth_max
                )
                for ref_pred_depth in ref_pred_depths_alpha_masked
            ],
            dim=0,
        )
        iio.mimwrite(
            osp.join(video_dir, "depths_alpha_masked.mp4"),
            make_video_divisble(
                (depth_alpha_masked_video.numpy() * 255).astype(np.uint8)
            ),
            fps=fps,
        )

        # surf normal video
        from flow3d.normal_utils import depth_to_normal
        # 确保 depth_min 和 depth_max 不相等，防止除零错误
        if abs(depth_max - depth_min) < 1e-6:
            guru.warning(f"depth_min ({depth_min}) and depth_max ({depth_max}) are too close, adjusting depth_max")
            depth_max = depth_min + 1e-3
        
        # 打印调试信息
        # print(f"\n[DEBUG] depth_to_normal parameters:")
        # print(f"  depth_min={depth_min}, depth_max={depth_max}")
        # print(f"  K[0,0] (fx)={K[0, 0].item()}, K[1,1] (fy)={K[1, 1].item()}")
        # print(f"  K[0,2] (cx)={K[0, 2].item()}, K[1,2] (cy)={K[1, 2].item()}")
        # print(f"  depth shape={ref_pred_depths[0].shape}")
        
        surf_normals = []
        for depth in ref_pred_depths:
            c2w = torch.linalg.inv(w2c)
            K_cpu = K
            surf_normal = depth_to_normal(depth[None].to(c2w.device), c2w[None], K_cpu[None], depth_min, depth_max)
            # import pdb
            # pdb.set_trace()
            surf_normal = (surf_normal * 0.5 + 0.5).cpu()
            surf_normal = (surf_normal - torch.min(surf_normal)) / (torch.max(surf_normal) - torch.min(surf_normal))
            surf_normals.append(surf_normal.squeeze(0))

        normal_video = torch.stack(surf_normals, dim=0)
        iio.mimwrite(
            osp.join(video_dir, "normals.mp4"),
            make_video_divisble((normal_video.numpy() * 255).astype(np.uint8)),
            fps=fps,
        )

        if len(normals) != 0:
            rend_normals = []
            for rend_n in normals:
                rend_n = (rend_n * 0.5 + 0.5)
                rend_n = (rend_n - torch.min(rend_n)) / (torch.max(rend_n) - torch.min(rend_n))
                rend_normals.append(rend_n.squeeze(0))
            rend_normal_video = torch.stack(rend_normals, dim=0)
            iio.mimwrite(
                osp.join(video_dir, "rend_normals.mp4"),
                make_video_divisble((rend_normal_video.numpy() * 255).astype(np.uint8)),
                fps=fps,
            )

            

        if len(masks) > 0:
            # mask video
            mask_video = torch.stack(masks, dim=0)
            iio.mimwrite(
                osp.join(video_dir, "masks.mp4"),
                make_video_divisble((mask_video.numpy() * 255).astype(np.uint8)),
                fps=fps,
            )

        # Render 2D track video.
        # tracks_2d, target_imgs = [], []
        # sample_interval = 10
        # batch0 = {
        #     k: v.to(self.device) if isinstance(v, torch.Tensor) else v
        #     for k, v in self.train_loader.dataset[0].items()
        # }
        # # ().
        # t = batch0["ts"]
        # # (4, 4).
        # w2c = batch0["w2cs"]
        # # (3, 3).
        # K = batch0["Ks"]
        # # (H, W, 3).
        # img = batch0["imgs"]
        # # (H, W).
        # bool_mask = batch0["masks"] > 0.5
        # img_wh = img.shape[-2::-1]
        # for batch in tqdm(
        #     self.train_loader, desc="Rendering 2D track video", leave=False
        # ):
        #     batch = {
        #         k: v.to(self.device) if isinstance(v, torch.Tensor) else v
        #         for k, v in batch.items()
        #     }
        #     # Putting results onto CPU since it will consume unnecessarily
        #     # large GPU memory for long sequence OW.
        #     # (1, H, W, 3).
        #     target_imgs.append(batch["imgs"].cpu())
        #     # (1,).
        #     target_ts = batch["ts"]
        #     # (1, 4, 4).
        #     target_w2cs = batch["w2cs"]
        #     # (1, 3, 3).
        #     target_Ks = batch["Ks"]
        #     rendered = self.model.render(
        #         t,
        #         w2c[None],
        #         K[None],
        #         img_wh,
        #         target_ts=target_ts,
        #         target_w2cs=target_w2cs,
        #     )
        #     pred_tracks_3d = rendered["tracks_3d"][0][
        #         ::sample_interval, ::sample_interval
        #     ][bool_mask[::sample_interval, ::sample_interval]].swapaxes(0, 1)
        #     pred_tracks_2d = torch.einsum("bij,bpj->bpi", target_Ks, pred_tracks_3d)
        #     pred_tracks_2d = pred_tracks_2d[..., :2] / torch.clamp(
        #         pred_tracks_2d[..., 2:], min=1e-6
        #     )
        #     tracks_2d.append(pred_tracks_2d.cpu())
        # tracks_2d = torch.cat(tracks_2d, dim=0)
        # target_imgs = torch.cat(target_imgs, dim=0)
        # track_2d_video = plot_correspondences(
        #     target_imgs.numpy(),
        #     tracks_2d.numpy(),
        #     query_id=cast(int, t),
        # )
        # iio.mimwrite(
        #     osp.join(video_dir, "tracks_2d.mp4"),
        #     make_video_divisble(np.stack(track_2d_video, 0)),
        #     fps=fps,
        # )
        
        # Render motion coefficient video.
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            coefs_for_video = self._get_motion_coefs_for_video()
            q = max(1, min(3, coefs_for_video.shape[0], coefs_for_video.shape[1]))
            motion_coef_colors = torch.pca_lowrank(
                coefs_for_video[None],
                q=q,
            )[0][0]
        if motion_coef_colors.shape[-1] < 3:
            motion_coef_colors = F.pad(motion_coef_colors, (0, 3 - motion_coef_colors.shape[-1]))
        color_min = motion_coef_colors.min(0)[0]
        color_range = (motion_coef_colors.max(0)[0] - color_min).clamp(min=1e-8)
        motion_coef_colors = (motion_coef_colors - color_min) / color_range
        motion_coef_colors = F.pad(
            motion_coef_colors, (0, 0, 0, self.model.num_bg_gaussians), value=0.5
        )
        assignment_confidence_colors = self._get_assignment_confidence_colors_for_video()
        assignment_confidence_colors = F.pad(
            assignment_confidence_colors,
            (0, 0, 0, self.model.num_bg_gaussians),
            value=0.0,
        )
        ghost_score_colors = self._get_ghost_score_colors_for_video()
        if ghost_score_colors is not None:
            ghost_score_colors = F.pad(
                ghost_score_colors,
                (0, 0, 0, self.model.num_bg_gaussians),
                value=0.0,
            )
        video = []
        confidence_video = []
        ghost_score_video = [] if ghost_score_colors is not None else None
        for batch in tqdm(
            self.train_loader, desc="Rendering motion coefficient video", leave=False
        ):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # ().
            t = batch["ts"][0]
            # (4, 4).
            w2c = batch["w2cs"][0]
            # (3, 3).
            K = batch["Ks"][0]
            # (3, 3).
            img = batch["imgs"][0]
            img_wh = img.shape[-2::-1]
            rendered = self.model.render(
                t, w2c[None], K[None], img_wh, colors_override=motion_coef_colors
            )
            confidence_rendered = self.model.render(
                t,
                w2c[None],
                K[None],
                img_wh,
                colors_override=assignment_confidence_colors,
            )
            if ghost_score_colors is not None:
                ghost_rendered = self.model.render(
                    t,
                    w2c[None],
                    K[None],
                    img_wh,
                    colors_override=ghost_score_colors,
                )
            # Putting results onto CPU since it will consume unnecessarily
            # large GPU memory for long sequence OW.
            video.append(torch.cat([img, rendered["img"][0]], dim=1).cpu())
            confidence_video.append(
                torch.cat([img, confidence_rendered["img"][0]], dim=1).cpu()
            )
            if ghost_score_video is not None:
                ghost_score_video.append(
                    torch.cat([img, ghost_rendered["img"][0]], dim=1).cpu()
                )
        video = torch.stack(video, dim=0)
        iio.mimwrite(
            osp.join(video_dir, "motion_coefs.mp4"),
            make_video_divisble((video.numpy() * 255).astype(np.uint8)),
            fps=fps,
        )
        confidence_video = torch.stack(confidence_video, dim=0)
        iio.mimwrite(
            osp.join(video_dir, "assignment_confidences.mp4"),
            make_video_divisble((confidence_video.numpy() * 255).astype(np.uint8)),
            fps=fps,
        )
        if ghost_score_video is not None:
            ghost_score_video = torch.stack(ghost_score_video, dim=0)
            iio.mimwrite(
                osp.join(video_dir, "ghost_scores.mp4"),
                make_video_divisble((ghost_score_video.numpy() * 255).astype(np.uint8)),
                fps=fps,
            )
