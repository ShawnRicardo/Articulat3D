import os
import os.path as osp
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
from flow3d.init_utils_articulat3d import (  # noqa: E402
    init_bg,
    init_fg_from_tracks_3d,
    init_motion_params_with_mask_ids_articulat3d,
    run_initial_optim_articulat3d,
)
from flow3d.scene_model import SceneModel  # noqa: E402
from flow3d.tensor_dataclass import StaticObservations, TrackObservations  # noqa: E402
from flow3d.trainer_articulat3d import (  # noqa: E402
    TrainerArticulat3D,
    disable_lpips_metric_downloads,
)
from flow3d.validator import Validator  # noqa: E402


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(42)


@dataclass
class TrainConfig:
    work_dir: str
    data: Articulat3DDataConfig
    lr: SceneLRConfig
    loss: LossesConfig
    optim: OptimizerConfig
    num_fg: int = 40_000
    num_bg: int = 0
    num_motion_bases: int | None = None
    num_epochs: int = 400
    port: int | None = 6006
    vis_debug: bool = False
    batch_size: int = 4
    num_dl_workers: int = 4
    validate_every: int = 50
    save_videos_every: int = 50
    use_2dgs: bool = False
    init_num_iters: int = 1000
    motion_segment_length: int = 200


def main(cfg: TrainConfig):
    disable_lpips_metric_downloads()
    os.makedirs(cfg.work_dir, exist_ok=True)
    backup_code(cfg.work_dir)

    train_dataset = Articulat3DDataset(**asdict(cfg.data))
    train_video_view = Articulat3DVideoView(train_dataset, include_static=True)
    guru.info(
        f"Training dataset samples={len(train_dataset)}, "
        f"motion_frames={train_dataset.num_frames}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(f"{cfg.work_dir}/cfg.yaml", "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False)

    ckpt_path = f"{cfg.work_dir}/checkpoints/last.ckpt"
    initialize_and_checkpoint_model(
        cfg=cfg,
        train_dataset=train_dataset,
        device=device,
        ckpt_path=ckpt_path,
    )

    trainer, start_epoch = TrainerArticulat3D.init_from_checkpoint(
        ckpt_path,
        device,
        cfg.use_2dgs,
        cfg.lr,
        cfg.loss,
        cfg.optim,
        work_dir=cfg.work_dir,
        port=cfg.port,
        motion_segment_length=cfg.motion_segment_length,
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
    )

    guru.info(f"Starting Articulat3D training from epoch {start_epoch}, global_step={trainer.global_step}")
    for epoch in (
        pbar := tqdm(
            range(start_epoch, cfg.num_epochs),
            initial=start_epoch,
            total=cfg.num_epochs,
            desc=f"Epoch {start_epoch}/{cfg.num_epochs - 1}",
        )
    ):
        trainer.set_epoch(epoch)
        for batch in train_loader:
            batch = to_device(batch, device)
            loss = trainer.train_step(batch)
            pbar.set_description(f"Loss: {loss:.6f}")

        if (epoch > 0 and epoch % cfg.validate_every == 0) or (epoch == cfg.num_epochs - 1):
            val_logs = validator.validate()
            if val_logs is not None:
                trainer.log_dict(val_logs)
        if (epoch > 0 and epoch % cfg.save_videos_every == 0) or (epoch == cfg.num_epochs - 1):
            validator.save_train_videos(epoch)


def initialize_and_checkpoint_model(
    cfg: TrainConfig,
    train_dataset: Articulat3DDataset,
    device: torch.device,
    ckpt_path: str,
):
    if os.path.exists(ckpt_path):
        guru.info(f"model checkpoint exists at {ckpt_path}")
        return

    fg_params, motion_bases, bg_params, tracks_3d = init_model_from_tracks(
        train_dataset=train_dataset,
        num_fg=cfg.num_fg,
        num_bg=cfg.num_bg,
        num_motion_bases=cfg.num_motion_bases,
        init_num_iters=cfg.init_num_iters,
        motion_segment_length=cfg.motion_segment_length,
    )

    model = SceneModel(
        Ks=train_dataset.get_Ks().to(device),
        w2cs=train_dataset.get_w2cs().to(device),
        fg_params=fg_params,
        motion_bases=motion_bases,
        camera_poses=None,
        bg_params=bg_params,
        use_2dgs=cfg.use_2dgs,
    )

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": 0, "global_step": 0}, ckpt_path)
    guru.info(f"Saved initialized Articulat3D model to {ckpt_path}")


def init_model_from_tracks(
    train_dataset: Articulat3DDataset,
    num_fg: int,
    num_bg: int,
    num_motion_bases: int | None,
    init_num_iters: int,
    motion_segment_length: int,
):
    tracks_3d = TrackObservations(*train_dataset.get_tracks_3d(num_fg))
    if not tracks_3d.check_sizes():
        raise RuntimeError("TrackObservations size check failed")
    if train_dataset.last_track_mask_ids is None:
        raise RuntimeError("Articulat3D dataset did not expose sampled mask_ids")

    cano_t = int(tracks_3d.visibles.sum(dim=0).argmax().item())
    guru.info(
        f"Canonical frame={cano_t}, fg={num_fg}, bg={num_bg}, "
        f"requested_bases={num_motion_bases}"
    )

    motion_bases, motion_coefs, tracks_3d = init_motion_params_with_mask_ids_articulat3d(
        tracks_3d=tracks_3d,
        mask_ids=train_dataset.last_track_mask_ids,
        num_bases=num_motion_bases,
        rot_type="6d",
        cano_t=cano_t,
        static_mask_id=0,
        motion_segment_length=motion_segment_length,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    motion_bases = motion_bases.to(device)
    fg_params = init_fg_from_tracks_3d(cano_t, tracks_3d, motion_coefs).to(device)
    tracks_3d = tracks_3d.to(device)

    if init_num_iters > 0:
        run_initial_optim_articulat3d(
            fg=fg_params,
            bases=motion_bases,
            tracks_3d=tracks_3d,
            Ks=train_dataset.get_dynamic_Ks().to(device),
            w2cs=train_dataset.get_dynamic_w2cs().to(device),
            num_iters=init_num_iters,
            use_depth_range_loss=True,
            motion_segment_length=motion_segment_length,
        )

    bg_params = None
    if num_bg > 0:
        bg_points = StaticObservations(*train_dataset.get_bkgd_points(num_bg))
        if not bg_points.check_sizes():
            raise RuntimeError("StaticObservations size check failed")
        bg_params = init_bg(bg_points).to(device)

    return fg_params, motion_bases, bg_params, tracks_3d


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
    tracks_3d_path = f"{data_dir}/trajectory_tapip3d_visualization.npz"
    work_dir = f"output/Articulat3DSimECCV/{scene_name}/stage1"

    cfg = TrainConfig(
        work_dir=work_dir,
        data=Articulat3DDataConfig(
            data_dir=data_dir,
            tracks_3d_path=tracks_3d_path,
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
        num_fg=40_000,
        num_bg=0,
        num_motion_bases=None,
        num_epochs=200,
        batch_size=16,
        num_dl_workers=4,
        port=6006,
        vis_debug=False,
        save_videos_every=50,
        validate_every=50,
        use_2dgs=False,
        init_num_iters=1000,
        motion_segment_length=200,
    )
    main(cfg)
