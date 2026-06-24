import os
import shutil
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from loguru import logger as guru
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

from flow3d.configs import (  # noqa: E402
    BGLRConfig,
    CameraPoseLRConfig,
    CameraScalesLRConfig,
    FGLRConfig,
    LossesConfig,
    MotionLRConfig,
    OptimizerConfig,
    SceneLRConfig,
)
from flow3d.data import BaseDataset  # noqa: E402
from flow3d.data.articulat3d_dataset import (  # noqa: E402
    Articulat3DDataConfig,
    Articulat3DDataset,
    Articulat3DVideoView,
)
from flow3d.data.utils import to_device  # noqa: E402
from flow3d.init_utils_articulat3d_stage2 import (  # noqa: E402
    initialize_stage2_model_from_stage1_checkpoint,
    load_gt_joint_viz_from_mobility,
)
from flow3d.articulated_motion_articulat3d_stage2 import (  # noqa: E402
    SceneModelArticulat3DStage2,
)
from flow3d.trainer_articulat3d_stage2 import TrainerArticulat3DStage2  # noqa: E402
from flow3d.trainer_articulat3d import disable_lpips_metric_downloads  # noqa: E402
from flow3d.validator import Validator  # noqa: E402
from flow3d.scene_model import SceneModel  # noqa: E402


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(42)


@dataclass
class Stage2TrainConfig:
    work_dir: str
    stage1_ckpt_path: str
    data: Articulat3DDataConfig
    lr: SceneLRConfig
    loss: LossesConfig
    optim: OptimizerConfig
    num_epochs: int = 200
    port: int | None = 6007
    batch_size: int = 4
    num_dl_workers: int = 4
    validate_every: int = 50
    save_videos_every: int = 50
    use_2dgs: bool = False
    motion_segment_length: int = 200
    joint_value_lr: float = 1.6e-4
    joint_axis_lr: float = 1.0e-5
    joint_pivot_lr: float = 1.0e-5
    unlock_joint_geometry_after_steps: int = 1000
    assignment_tau_init: float = 1.0
    assignment_tau_final: float = 0.1
    assignment_tau_decay_steps: int = 5000
    use_assignment_pseudo_targets: bool = False
    w_assignment_prior: float = 0.0
    w_assignment_spatial_prior: float = 0.0
    w_assignment_graph_smooth: float = 0.0
    w_assignment_entropy: float = 0.0
    use_stochastic_assignment: bool = False
    w_axis_prior: float = 1.0e-3
    w_pivot_prior: float = 1.0e-3
    unlock_all_assignment_epoch: int = 50
    unlock_opacity_epoch: int = 50
    unlock_means_epoch: int = 100
    stage2_opacity_lr: float = 1.0e-3
    stage2_means_lr: float = 1.0e-5
    w_opacity_anchor: float = 1.0e-4
    w_means_anchor: float = 1.0e-2
    train_stage2_colors: bool = False
    train_stage2_quats: bool = False
    train_stage2_scales: bool = False
    assignment_logit_strength: float = 4.0
    disable_gaussian_control: bool = True
    w_outside_alpha: float = 0.5
    outside_mask_dilate_px: int = 3
    enable_ghost_score: bool = True
    ghost_start_epoch: int = 50
    ghost_score_update_every_steps: int = 20
    ghost_min_visible: int = 5
    ghost_depth_abs_thresh: float = 0.05
    ghost_depth_rel_thresh: float = 0.03
    ghost_opacity_score_thresh: float = 0.6
    w_ghost_opacity_decay: float = 5.0e-3
    enable_stage2_ghost_cull: bool = True
    ghost_cull_every_steps: int = 500
    ghost_cull_score_thresh: float = 0.75
    ghost_cull_opacity_thresh: float = 0.08
    max_ghost_cull_per_step: int = 2000
    enable_direct_reassignment: bool = True
    assignment_reassign_start_epoch: int = 50
    assignment_reassign_every_steps: int = 500
    max_reassign_per_step: int = 3000
    reassign_candidate_score_thresh: float = 0.6
    reassign_min_visible: int = 5
    reassign_min_opacity: float = 0.05
    reassign_logit_strength: float = 4.0
    reassign_improve_ratio: float = 0.20
    w_reassign_mask: float = 1.0
    w_reassign_depth: float = 2.0
    w_reassign_part_prior: float = 0.0
    w_reassign_knn: float = 0.1
    reassign_switch_penalty: float = 0.05
    part_prior_min_points: int = 128
    part_prior_conf_thresh: float = -1.0
    part_prior_opacity_thresh: float = 0.1
    part_prior_ghost_score_thresh: float = 0.5
    part_prior_scale_min: float = 0.03
    part_prior_center_mode: str = "robust_obb"
    part_prior_obb_low_quantile: float = 0.05
    part_prior_obb_high_quantile: float = 0.95
    ghost_cull_requires_reassign_attempt: bool = True
    depth_alpha_vis_thresh: float = 0.15
    show_joint_axes: bool = True
    show_gt_joint_axes: bool = True
    viewer_joint_update_every: int = 100
    joint_axis_init_line_width: float = 2.0
    joint_axis_current_line_width: float = 4.0
    gt_joint_axis_line_width: float = 3.0
    joint_pivot_point_size: float = 0.035
    show_part_prior_centers: bool = False
    part_prior_center_point_size: float = 0.08
    viewer_part_centers_follow_motion: bool = True
    viewer_render_uses_timestep: bool = True
    gt_joint_axis_path: str | None = None
    freeze_gaussian_params: bool = False
    train_axis_pivot: bool = False
    train_assignment_conf_thresh: float = 0.8
    fit_core_min_conf: float = 0.7
    fit_trim_quantile: float = 0.8
    enable_assignment_refine: bool = True
    assignment_refine_stride: int = 5
    assignment_residual_bad_quantile: float = 0.8
    assignment_switch_improve_ratio: float = 0.15
    enable_assignment_knn_prior: bool = True
    spatial_knn_k: int = 24
    spatial_majority_ratio: float = 0.70
    spatial_residual_slack_ratio: float = 0.10
    assignment_refine_chunk_size: int = 2048
    assignment_graph_stride: int = 50
    assignment_graph_motion_quantile: float = 0.8
    init_gate_psnr_drop_db: float = 5.0
    init_gate_stride: int = 10


def main(cfg: Stage2TrainConfig):
    disable_lpips_metric_downloads()
    os.makedirs(cfg.work_dir, exist_ok=True)
    backup_code(cfg.work_dir)

    train_dataset = Articulat3DDataset(**asdict(cfg.data))
    train_video_view = Articulat3DVideoView(train_dataset, include_static=True)
    guru.info(
        f"Stage2 dataset samples={len(train_dataset)}, "
        f"motion_frames={train_dataset.num_frames}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(f"{cfg.work_dir}/cfg.yaml", "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False)

    ckpt_path = f"{cfg.work_dir}/checkpoints/last.ckpt"
    initialize_and_checkpoint_stage2(cfg, device, ckpt_path, train_dataset)

    trainer, start_epoch = TrainerArticulat3DStage2.init_from_checkpoint(
        ckpt_path,
        device,
        cfg.use_2dgs,
        cfg.lr,
        cfg.loss,
        cfg.optim,
        work_dir=cfg.work_dir,
        port=cfg.port,
        motion_segment_length=cfg.motion_segment_length,
        joint_value_lr=cfg.joint_value_lr,
        joint_axis_lr=cfg.joint_axis_lr,
        joint_pivot_lr=cfg.joint_pivot_lr,
        unlock_joint_geometry_after_steps=cfg.unlock_joint_geometry_after_steps,
        assignment_tau_init=cfg.assignment_tau_init,
        assignment_tau_final=cfg.assignment_tau_final,
        assignment_tau_decay_steps=cfg.assignment_tau_decay_steps,
        use_assignment_pseudo_targets=cfg.use_assignment_pseudo_targets,
        w_assignment_prior=cfg.w_assignment_prior,
        w_assignment_spatial_prior=cfg.w_assignment_spatial_prior,
        w_assignment_graph_smooth=cfg.w_assignment_graph_smooth,
        w_assignment_entropy=cfg.w_assignment_entropy,
        use_stochastic_assignment=cfg.use_stochastic_assignment,
        w_axis_prior=cfg.w_axis_prior,
        w_pivot_prior=cfg.w_pivot_prior,
        unlock_all_assignment_epoch=cfg.unlock_all_assignment_epoch,
        unlock_opacity_epoch=cfg.unlock_opacity_epoch,
        unlock_means_epoch=cfg.unlock_means_epoch,
        stage2_opacity_lr=cfg.stage2_opacity_lr,
        stage2_means_lr=cfg.stage2_means_lr,
        w_opacity_anchor=cfg.w_opacity_anchor,
        w_means_anchor=cfg.w_means_anchor,
        train_stage2_colors=cfg.train_stage2_colors,
        train_stage2_quats=cfg.train_stage2_quats,
        train_stage2_scales=cfg.train_stage2_scales,
        disable_gaussian_control=cfg.disable_gaussian_control,
        w_outside_alpha=cfg.w_outside_alpha,
        outside_mask_dilate_px=cfg.outside_mask_dilate_px,
        enable_ghost_score=cfg.enable_ghost_score,
        ghost_start_epoch=cfg.ghost_start_epoch,
        ghost_score_update_every_steps=cfg.ghost_score_update_every_steps,
        ghost_min_visible=cfg.ghost_min_visible,
        ghost_depth_abs_thresh=cfg.ghost_depth_abs_thresh,
        ghost_depth_rel_thresh=cfg.ghost_depth_rel_thresh,
        ghost_opacity_score_thresh=cfg.ghost_opacity_score_thresh,
        w_ghost_opacity_decay=cfg.w_ghost_opacity_decay,
        enable_stage2_ghost_cull=cfg.enable_stage2_ghost_cull,
        ghost_cull_every_steps=cfg.ghost_cull_every_steps,
        ghost_cull_score_thresh=cfg.ghost_cull_score_thresh,
        ghost_cull_opacity_thresh=cfg.ghost_cull_opacity_thresh,
        max_ghost_cull_per_step=cfg.max_ghost_cull_per_step,
        enable_direct_reassignment=cfg.enable_direct_reassignment,
        assignment_reassign_start_epoch=cfg.assignment_reassign_start_epoch,
        assignment_reassign_every_steps=cfg.assignment_reassign_every_steps,
        max_reassign_per_step=cfg.max_reassign_per_step,
        reassign_candidate_score_thresh=cfg.reassign_candidate_score_thresh,
        reassign_min_visible=cfg.reassign_min_visible,
        reassign_min_opacity=cfg.reassign_min_opacity,
        reassign_logit_strength=cfg.reassign_logit_strength,
        reassign_improve_ratio=cfg.reassign_improve_ratio,
        w_reassign_mask=cfg.w_reassign_mask,
        w_reassign_depth=cfg.w_reassign_depth,
        w_reassign_part_prior=cfg.w_reassign_part_prior,
        w_reassign_knn=cfg.w_reassign_knn,
        reassign_switch_penalty=cfg.reassign_switch_penalty,
        part_prior_min_points=cfg.part_prior_min_points,
        part_prior_conf_thresh=cfg.part_prior_conf_thresh,
        part_prior_opacity_thresh=cfg.part_prior_opacity_thresh,
        part_prior_ghost_score_thresh=cfg.part_prior_ghost_score_thresh,
        part_prior_scale_min=cfg.part_prior_scale_min,
        part_prior_center_mode=cfg.part_prior_center_mode,
        part_prior_obb_low_quantile=cfg.part_prior_obb_low_quantile,
        part_prior_obb_high_quantile=cfg.part_prior_obb_high_quantile,
        ghost_cull_requires_reassign_attempt=cfg.ghost_cull_requires_reassign_attempt,
        show_joint_axes=cfg.show_joint_axes,
        show_gt_joint_axes=cfg.show_gt_joint_axes,
        viewer_joint_update_every=cfg.viewer_joint_update_every,
        joint_axis_init_line_width=cfg.joint_axis_init_line_width,
        joint_axis_current_line_width=cfg.joint_axis_current_line_width,
        gt_joint_axis_line_width=cfg.gt_joint_axis_line_width,
        joint_pivot_point_size=cfg.joint_pivot_point_size,
        show_part_prior_centers=cfg.show_part_prior_centers,
        part_prior_center_point_size=cfg.part_prior_center_point_size,
        viewer_part_centers_follow_motion=cfg.viewer_part_centers_follow_motion,
        viewer_render_uses_timestep=cfg.viewer_render_uses_timestep,
        freeze_gaussian_params=cfg.freeze_gaussian_params,
        train_axis_pivot=cfg.train_axis_pivot,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_dl_workers,
        persistent_workers=cfg.num_dl_workers > 0,
        collate_fn=BaseDataset.train_collate_fn,
    )
    validator = Validator(
        model=trainer.model,
        device=device,
        train_loader=DataLoader(train_video_view, batch_size=1),
        val_img_loader=None,
        val_kpt_loader=None,
        save_dir=cfg.work_dir,
        depth_alpha_vis_thresh=cfg.depth_alpha_vis_thresh,
    )

    guru.info(f"Starting Articulat3D stage2 from epoch {start_epoch}, global_step={trainer.global_step}")
    for epoch in (
        pbar := tqdm(
            range(start_epoch, cfg.num_epochs),
            initial=start_epoch,
            total=cfg.num_epochs,
            desc=f"Stage2 Epoch {start_epoch}/{cfg.num_epochs - 1}",
        )
    ):
        trainer.set_epoch(epoch)
        for batch in train_loader:
            batch = to_device(batch, device)
            loss = trainer.train_step(batch)
            pbar.set_description(f"Stage2 Loss: {loss:.6f}")

        if (epoch > 0 and epoch % cfg.validate_every == 0) or (epoch == cfg.num_epochs - 1):
            val_logs = validator.validate()
            if val_logs is not None:
                trainer.log_dict(val_logs)
        if (epoch > 0 and epoch % cfg.save_videos_every == 0) or (epoch == cfg.num_epochs - 1):
            validator.save_train_videos(epoch)


def initialize_and_checkpoint_stage2(
    cfg: Stage2TrainConfig,
    device: torch.device,
    ckpt_path: str,
    train_dataset: Articulat3DDataset,
):
    if os.path.exists(ckpt_path):
        guru.info(f"stage2 checkpoint exists at {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
            model_state = ckpt.get("model", {})
            required_stage2_keys = {
                "assignment_graph_edges",
                "assignment_graph_weights",
                "initial_fg_means",
                "initial_fg_opacities",
                "opacity_train_mask",
                "means_train_mask",
            }
            missing = sorted(required_stage2_keys - set(model_state))
            if missing:
                guru.warning(
                    "Existing stage2 checkpoint predates target-free graph/gaussian "
                    f"training buffers and is missing {missing}. Initialization is "
                    "skipped. Use a new work_dir or move the old checkpoint if you "
                    "want the new stage2 logic."
                )
            gt_keys = {
                "motion_bases.gt_prismatic_segments",
                "motion_bases.gt_revolute_segments",
                "motion_bases.gt_revolute_pivots",
            }
            missing_gt_keys = sorted(gt_keys - set(model_state))
            if cfg.show_gt_joint_axes and missing_gt_keys:
                model = SceneModelArticulat3DStage2.init_from_stage2_state_dict(model_state)
                (
                    gt_prismatic_segments,
                    gt_revolute_segments,
                    gt_revolute_pivots,
                    gt_joint_viz_report,
                ) = load_gt_joint_viz_from_mobility(
                    scene_path=cfg.data.data_dir,
                    bases=model.motion_bases,
                    scene_norm_transform=train_dataset.scene_norm_transform,
                    scene_norm_scale=train_dataset.scene_norm_scale,
                    gt_joint_axis_path=cfg.gt_joint_axis_path,
                )
                model.motion_bases.set_gt_joint_viz(
                    gt_prismatic_segments=gt_prismatic_segments,
                    gt_revolute_segments=gt_revolute_segments,
                    gt_revolute_pivots=gt_revolute_pivots,
                )
                ckpt["model"] = model.state_dict()
                torch.save(ckpt, ckpt_path)
                guru.info(
                    "Patched existing stage2 checkpoint with GT joint viewer buffers: "
                    f"{gt_joint_viz_report}"
                )
        except Exception as exc:
            guru.warning(f"Could not inspect existing stage2 checkpoint: {exc}")
        return
    model, init_report = initialize_stage2_model_from_stage1_checkpoint(
        stage1_ckpt_path=cfg.stage1_ckpt_path,
        scene_path=cfg.data.data_dir,
        device=device,
        use_2dgs=cfg.use_2dgs,
        assignment_logit_strength=cfg.assignment_logit_strength,
        scene_norm_transform=train_dataset.scene_norm_transform,
        scene_norm_scale=train_dataset.scene_norm_scale,
        assignment_train_conf_thresh=cfg.train_assignment_conf_thresh,
        fit_core_min_conf=cfg.fit_core_min_conf,
        fit_trim_quantile=cfg.fit_trim_quantile,
        enable_assignment_refine=cfg.enable_assignment_refine,
        assignment_refine_stride=cfg.assignment_refine_stride,
        assignment_residual_bad_quantile=cfg.assignment_residual_bad_quantile,
        assignment_switch_improve_ratio=cfg.assignment_switch_improve_ratio,
        enable_assignment_knn_prior=cfg.enable_assignment_knn_prior,
        spatial_knn_k=cfg.spatial_knn_k,
        spatial_majority_ratio=cfg.spatial_majority_ratio,
        spatial_residual_slack_ratio=cfg.spatial_residual_slack_ratio,
        assignment_refine_chunk_size=cfg.assignment_refine_chunk_size,
        assignment_graph_stride=cfg.assignment_graph_stride,
        assignment_graph_motion_quantile=cfg.assignment_graph_motion_quantile,
        gt_joint_axis_path=cfg.gt_joint_axis_path,
    )
    gate_report = run_init_psnr_gate(cfg, train_dataset, model, device)
    init_report["init_psnr_gate"] = gate_report
    if "assignment_diagnostics" in init_report:
        with open(Path(cfg.work_dir) / "stage2_assignment_diagnostics.yaml", "w") as f:
            yaml.dump(init_report["assignment_diagnostics"], f, default_flow_style=False)
    with open(Path(cfg.work_dir) / "stage2_init_report.yaml", "w") as f:
        yaml.dump(init_report, f, default_flow_style=False)
    if not gate_report["passed"]:
        raise RuntimeError(
            "Stage2 init PSNR gate failed: "
            f"stage1={gate_report['stage1_mean_psnr']:.3f}, "
            f"stage2={gate_report['stage2_mean_psnr']:.3f}, "
            f"drop={gate_report['psnr_drop_db']:.3f} dB"
        )
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": 0,
            "global_step": 0,
            "stage2_init_report": init_report,
        },
        ckpt_path,
    )
    guru.info(f"Saved initialized stage2 model to {ckpt_path}")


@torch.inference_mode()
def run_init_psnr_gate(
    cfg: Stage2TrainConfig,
    train_dataset: Articulat3DDataset,
    stage2_model,
    device: torch.device,
) -> dict:
    stage1_ckpt = torch.load(cfg.stage1_ckpt_path, weights_only=False, map_location=device)
    stage1_model = SceneModel.init_from_state_dict(stage1_ckpt["model"]).to(device)
    stage1_model.use_2dgs = cfg.use_2dgs
    stage2_model.use_2dgs = cfg.use_2dgs
    stage1_model.training = False
    stage2_model.training = False

    img_wh = train_dataset.get_img_wh()
    frame_indices = list(range(0, train_dataset.num_frames, cfg.init_gate_stride))
    rows = []
    for frame_idx in frame_indices:
        img = train_dataset.get_image(frame_idx).to(device)
        mask = (train_dataset.get_mask(frame_idx).to(device) > 0).float()
        w2c = train_dataset.w2cs[frame_idx].to(device)
        K = train_dataset.Ks[frame_idx].to(device)
        target = img * mask[..., None] + (1.0 - mask[..., None])
        stage1_img = stage1_model.render(
            frame_idx, w2c[None], K[None], img_wh, bg_color=1.0
        )["img"][0]
        stage2_img = stage2_model.render(
            frame_idx, w2c[None], K[None], img_wh, bg_color=1.0
        )["img"][0]
        stage1_psnr = masked_psnr(stage1_img, target, mask)
        stage2_psnr = masked_psnr(stage2_img, target, mask)
        rows.append((frame_idx, stage1_psnr, stage2_psnr))

    stage1_mean = float(np.mean([r[1] for r in rows]))
    stage2_mean = float(np.mean([r[2] for r in rows]))
    drop = stage1_mean - stage2_mean
    passed = drop <= cfg.init_gate_psnr_drop_db
    metrics_path = Path(cfg.work_dir) / "stage2_init_gate_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("frame\tstage1_psnr\tstage2_psnr\tdrop_db\n")
        for frame_idx, stage1_psnr, stage2_psnr in rows:
            f.write(
                f"{frame_idx:06d}\t{stage1_psnr:.6f}\t{stage2_psnr:.6f}\t"
                f"{stage1_psnr - stage2_psnr:.6f}\n"
            )
        f.write(f"\nmean\t{stage1_mean:.6f}\t{stage2_mean:.6f}\t{drop:.6f}\n")
        f.write(f"passed\t{passed}\n")
    return {
        "passed": bool(passed),
        "stage1_mean_psnr": stage1_mean,
        "stage2_mean_psnr": stage2_mean,
        "psnr_drop_db": drop,
        "threshold_db": cfg.init_gate_psnr_drop_db,
        "stride": cfg.init_gate_stride,
        "metrics_path": str(metrics_path),
    }


def masked_psnr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    weight = mask[..., None].to(dtype=pred.dtype)
    denom = (weight.sum() * pred.shape[-1]).clamp_min(1.0)
    mse = (((pred - target) ** 2) * weight).sum() / denom
    return float((-10.0 * torch.log10(mse.clamp_min(1e-10))).item())


def backup_code(work_dir: str):
    repo_root = Path(__file__).resolve().parents[1]
    dst_dir = Path(work_dir) / "code" / datetime.now().strftime("%Y-%m-%d-%H%M%S")
    dst_dir.mkdir(parents=True, exist_ok=True)
    for dirname in ["flow3d", "scripts", "preprocess"]:
        src = repo_root / dirname
        if src.exists():
            shutil.copytree(src, dst_dir / dirname, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


if __name__ == "__main__":
    scene_name = "StorageFurniture_45194"
    data_dir = f"data/Articulat3DSimECCV/{scene_name}"
    scene_work_dir = f"output/Articulat3DSimECCV/{scene_name}"
    stage1_work_dir = f"{scene_work_dir}/stage1"
    stage1_ckpt_path = f"{stage1_work_dir}/checkpoints/last.ckpt"
    work_dir = f"{scene_work_dir}/stage2"

    cfg = Stage2TrainConfig(
        work_dir=work_dir,
        stage1_ckpt_path=stage1_ckpt_path,
        data=Articulat3DDataConfig(
            data_dir=data_dir,
            tracks_3d_path=f"{data_dir}/trajectory_tapip3d_visualization.npz",
            target_size=512,
            depth_scale=6553.5,
            motion_num_frames=600,
            track_phase_frames=200,
            dynamic_start=0,
            dynamic_end=600,
            static_start=600,
            static_end=750,
            static_motion_ts=0,
        ),
        lr=SceneLRConfig(
            fg=FGLRConfig(),
            bg=BGLRConfig(),
            motion_bases=MotionLRConfig(),
            camera_poses=CameraPoseLRConfig(),
            camera_scales=CameraScalesLRConfig(),
        ),
        loss=LossesConfig(),
        optim=OptimizerConfig(),
        num_epochs=200,
        batch_size=16,
        num_dl_workers=4,
        port=6007,
        validate_every=50,
        save_videos_every=50,
        use_2dgs=False,
        motion_segment_length=200,
        joint_value_lr=1.6e-4,
        joint_axis_lr=1.0e-5,
        joint_pivot_lr=1.0e-5,
        unlock_joint_geometry_after_steps=1000,
        assignment_tau_init=1.0,
        assignment_tau_final=0.1,
        assignment_tau_decay_steps=5000,
        use_assignment_pseudo_targets=False,
        w_assignment_prior=0.0,
        w_assignment_spatial_prior=0.0,
        w_assignment_graph_smooth=0.0,
        w_assignment_entropy=0.0,
        use_stochastic_assignment=False,
        w_axis_prior=1.0e-3,
        w_pivot_prior=1.0e-3,
        unlock_all_assignment_epoch=50,
        unlock_opacity_epoch=50,
        unlock_means_epoch=100,
        stage2_opacity_lr=1.0e-3,
        stage2_means_lr=1.0e-5,
        w_opacity_anchor=1.0e-4,
        w_means_anchor=1.0e-2,
        train_stage2_colors=False,
        train_stage2_quats=False,
        train_stage2_scales=False,
        assignment_logit_strength=4.0,
        disable_gaussian_control=True,
        w_outside_alpha=0.5,
        outside_mask_dilate_px=3,
        enable_ghost_score=True,
        ghost_start_epoch=50,
        ghost_score_update_every_steps=20,
        ghost_min_visible=5,
        ghost_depth_abs_thresh=0.05,
        ghost_depth_rel_thresh=0.03,
        ghost_opacity_score_thresh=0.6,
        w_ghost_opacity_decay=5.0e-3,
        enable_stage2_ghost_cull=True,
        ghost_cull_every_steps=500,
        ghost_cull_score_thresh=0.75,
        ghost_cull_opacity_thresh=0.08,
        max_ghost_cull_per_step=2000,
        enable_direct_reassignment=True,
        assignment_reassign_start_epoch=50,
        assignment_reassign_every_steps=500,
        max_reassign_per_step=3000,
        reassign_candidate_score_thresh=0.6,
        reassign_min_visible=5,
        reassign_min_opacity=0.05,
        reassign_logit_strength=4.0,
        reassign_improve_ratio=0.20,
        w_reassign_mask=1.0,
        w_reassign_depth=2.0,
        w_reassign_part_prior=0.0,
        w_reassign_knn=0.1,
        reassign_switch_penalty=0.05,
        part_prior_min_points=128,
        part_prior_conf_thresh=-1.0,
        part_prior_opacity_thresh=0.1,
        part_prior_ghost_score_thresh=0.5,
        part_prior_scale_min=0.03,
        part_prior_center_mode="robust_obb",
        part_prior_obb_low_quantile=0.05,
        part_prior_obb_high_quantile=0.95,
        ghost_cull_requires_reassign_attempt=True,
        depth_alpha_vis_thresh=0.15,
        show_joint_axes=True,
        show_gt_joint_axes=True,
        viewer_joint_update_every=100,
        joint_axis_init_line_width=2.0,
        joint_axis_current_line_width=4.0,
        gt_joint_axis_line_width=3.0,
        joint_pivot_point_size=0.035,
        show_part_prior_centers=False,
        part_prior_center_point_size=0.08,
        viewer_part_centers_follow_motion=True,
        viewer_render_uses_timestep=True,
        gt_joint_axis_path=None,
        freeze_gaussian_params=False,
        train_axis_pivot=False,
        train_assignment_conf_thresh=0.8,
        fit_core_min_conf=0.7,
        fit_trim_quantile=0.8,
        enable_assignment_refine=True,
        assignment_refine_stride=5,
        assignment_residual_bad_quantile=0.8,
        assignment_switch_improve_ratio=0.15,
        enable_assignment_knn_prior=True,
        spatial_knn_k=24,
        spatial_majority_ratio=0.70,
        spatial_residual_slack_ratio=0.10,
        assignment_refine_chunk_size=2048,
        assignment_graph_stride=50,
        assignment_graph_motion_quantile=0.8,
        init_gate_psnr_drop_db=5.0,
        init_gate_stride=10,
    )
    main(cfg)
