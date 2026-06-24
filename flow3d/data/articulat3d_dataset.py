import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import roma
import torch
import torch.nn.functional as F
from loguru import logger as guru
from PIL import Image
from torch.utils.data import Dataset

from flow3d.data.base_dataset import BaseDataset
from flow3d.transforms import rt_to_mat4


@dataclass
class Articulat3DDataConfig:
    data_dir: str
    tracks_3d_path: str
    target_size: int = 512
    depth_scale: float = 6553.5
    motion_num_frames: int = 600
    track_phase_frames: int = 200
    dynamic_start: int = 0
    dynamic_end: int = 600
    static_start: int = 600
    static_end: int = 750
    static_motion_ts: int = 0


def _frame_name(frame_id: int) -> str:
    return f"{frame_id:06d}"


def _resize_rgb(rgb: np.ndarray, size: int) -> torch.Tensor:
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(img).copy()).float() / 255.0


def _resize_float_image(arr: np.ndarray, size: int, mode: str) -> torch.Tensor:
    ten = torch.from_numpy(arr.astype(np.float32))[None, None]
    out = F.interpolate(ten, size=(size, size), mode=mode, align_corners=False if mode != "nearest" else None)
    return out[0, 0].float()


def _load_intrinsics(scene_path: Path) -> np.ndarray:
    with open(scene_path / "intrinsics.json", "r") as f:
        K = np.array(json.load(f), dtype=np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"{scene_path / 'intrinsics.json'} must contain a 3x3 matrix, got {K.shape}")
    return K


def _load_camera_poses(scene_path: Path) -> dict[str, list]:
    with open(scene_path / "camera_pose.json", "r") as f:
        poses = json.load(f)
    if not isinstance(poses, dict):
        raise ValueError(f"{scene_path / 'camera_pose.json'} must be keyed by frame id")
    return poses


def _compute_scene_norm(points: torch.Tensor, w2cs: torch.Tensor) -> tuple[float, torch.Tensor]:
    points = points.reshape(-1, 3)
    valid = torch.isfinite(points).all(dim=-1)
    points = points[valid]
    if points.numel() == 0:
        raise ValueError("No finite 3D points available for scene normalization")

    scene_center = points.mean(dim=0)
    centered = points - scene_center[None]
    min_scale = centered.quantile(0.05, dim=0)
    max_scale = centered.quantile(0.95, dim=0)
    scale = (max_scale - min_scale).max().item() / 2.0
    if scale <= 0:
        raise ValueError(f"Invalid scene normalization scale: {scale}")

    original_up = -F.normalize(w2cs[:, 1, :3].mean(0), dim=-1)
    target_up = original_up.new_tensor([0.0, 0.0, 1.0])
    dot = original_up.dot(target_up).clamp(-1.0, 1.0)
    cross = original_up.cross(target_up, dim=-1)
    if cross.norm() < 1e-6:
        R = torch.eye(3, dtype=points.dtype, device=points.device)
    else:
        R = roma.rotvec_to_rotmat(F.normalize(cross, dim=-1) * dot.acos())
    transform = rt_to_mat4(R, torch.einsum("ij,j->i", -R, scene_center))
    return scale, transform


class Articulat3DDataset(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        tracks_3d_path: str,
        target_size: int = 512,
        depth_scale: float = 6553.5,
        motion_num_frames: int = 600,
        track_phase_frames: int = 200,
        dynamic_start: int = 0,
        dynamic_end: int = 600,
        static_start: int = 600,
        static_end: int = 750,
        static_motion_ts: int = 0,
        **_,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.tracks_3d_path = Path(tracks_3d_path)
        self.target_size = target_size
        self.depth_scale = depth_scale
        self.motion_num_frames = motion_num_frames
        self.track_phase_frames = track_phase_frames
        self.dynamic_start = dynamic_start
        self.dynamic_end = dynamic_end
        self.static_start = static_start
        self.static_end = static_end
        self.static_motion_ts = static_motion_ts

        if not self.data_dir.exists():
            raise FileNotFoundError(self.data_dir)
        if not self.tracks_3d_path.exists():
            raise FileNotFoundError(self.tracks_3d_path)
        if motion_num_frames % track_phase_frames != 0:
            raise ValueError(
                f"motion_num_frames={motion_num_frames} must be divisible by "
                f"track_phase_frames={track_phase_frames}"
            )
        if dynamic_end - dynamic_start != motion_num_frames:
            raise ValueError(
                f"dynamic range [{dynamic_start}, {dynamic_end}) must contain "
                f"{motion_num_frames} frames"
            )

        self.frame_ids = list(range(dynamic_start, dynamic_end)) + list(range(static_start, static_end))
        self.frame_names = [_frame_name(i) for i in self.frame_ids]
        self._ts = [
            i - dynamic_start if dynamic_start <= i < dynamic_end else static_motion_ts
            for i in self.frame_ids
        ]
        if max(self._ts) >= motion_num_frames or min(self._ts) < 0:
            raise ValueError(f"Invalid motion timestamps for {motion_num_frames} motion frames")

        self.imgs: list[torch.Tensor | None] = [None for _ in self.frame_ids]
        self.depths: list[torch.Tensor | None] = [None for _ in self.frame_ids]
        self.masks: list[torch.Tensor | None] = [None for _ in self.frame_ids]

        self.raw_w2cs, self.Ks = self._load_cameras()
        self.scene_norm_scale, self.scene_norm_transform = self._build_scene_norm()
        self.w2cs = torch.einsum(
            "nij,jk->nik", self.raw_w2cs, torch.linalg.inv(self.scene_norm_transform)
        )
        self.w2cs[:, :3, 3] /= self.scene_norm_scale

        self.last_track_mask_ids: torch.Tensor | None = None
        self._tracks_cache: dict[tuple[int, int, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        guru.info(
            f"Loaded Articulat3D dataset: frames={len(self.frame_ids)}, "
            f"motion_frames={self.motion_num_frames}, target_size={self.target_size}, "
            f"scene_scale={self.scene_norm_scale:.6f}"
        )

    @property
    def num_frames(self) -> int:
        return self.motion_num_frames

    @property
    def keyframe_idcs(self) -> torch.Tensor:
        return torch.arange(self.motion_num_frames)

    def __len__(self) -> int:
        return len(self.frame_ids)

    def _load_cameras(self) -> tuple[torch.Tensor, torch.Tensor]:
        camera_poses = _load_camera_poses(self.data_dir)
        K = _load_intrinsics(self.data_dir)

        first_img = Image.open(self.data_dir / "images" / f"{self.frame_names[0]}.png")
        src_w, src_h = first_img.size
        sx = self.target_size / float(src_w)
        sy = self.target_size / float(src_h)
        K_scaled = K.copy()
        K_scaled[0, 0] *= sx
        K_scaled[0, 2] *= sx
        K_scaled[1, 1] *= sy
        K_scaled[1, 2] *= sy

        c2ws = []
        for name in self.frame_names:
            if name not in camera_poses:
                raise KeyError(f"Missing camera pose for frame {name}")
            c2w = np.array(camera_poses[name], dtype=np.float32)
            c2w[:3, :3] = c2w[:3, :3] @ np.diag([1, -1, -1]).astype(np.float32)
            c2ws.append(c2w)

        c2ws_np = np.stack(c2ws, axis=0)
        w2cs = torch.from_numpy(np.linalg.inv(c2ws_np)).float()
        Ks = torch.from_numpy(np.repeat(K_scaled[None], len(self.frame_names), axis=0)).float()
        return w2cs, Ks

    def _load_track_arrays(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = np.load(self.tracks_3d_path)
        required = {"coords", "visibs", "mask_ids"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{self.tracks_3d_path} is missing required keys: {sorted(missing)}")
        coords = torch.from_numpy(data["coords"]).float()
        visibs = torch.from_numpy(data["visibs"]).bool()
        mask_ids = torch.from_numpy(data["mask_ids"]).long()

        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError(f"coords must have shape (T,N,3), got {tuple(coords.shape)}")
        if visibs.shape != coords.shape[:2]:
            raise ValueError(f"visibs shape {tuple(visibs.shape)} does not match coords {tuple(coords.shape[:2])}")
        if mask_ids.shape[0] != coords.shape[1]:
            raise ValueError(f"mask_ids length {mask_ids.shape[0]} does not match track count {coords.shape[1]}")
        if coords.shape[0] != self.track_phase_frames:
            raise ValueError(
                f"Expected {self.track_phase_frames} phase track frames, got {coords.shape[0]}"
            )
        return coords, visibs, mask_ids

    def _build_scene_norm(self) -> tuple[float, torch.Tensor]:
        coords, visibs, _ = self._load_track_arrays()
        valid_coords = coords[visibs]
        return _compute_scene_norm(valid_coords, self.raw_w2cs[: self.motion_num_frames])

    def _normalize_points(self, xyz: torch.Tensor) -> torch.Tensor:
        flat = xyz.reshape(-1, 3)
        flat = torch.einsum(
            "ij,nj->ni",
            self.scene_norm_transform[:3, :3],
            flat,
        ) + self.scene_norm_transform[:3, 3]
        return (flat / self.scene_norm_scale).reshape_as(xyz)

    def get_w2cs(self) -> torch.Tensor:
        return self.w2cs

    def get_dynamic_w2cs(self) -> torch.Tensor:
        return self.w2cs[: self.motion_num_frames]

    def get_Ks(self) -> torch.Tensor:
        return self.Ks

    def get_dynamic_Ks(self) -> torch.Tensor:
        return self.Ks[: self.motion_num_frames]

    def get_img_wh(self) -> tuple[int, int]:
        return self.target_size, self.target_size

    def get_image(self, index: int) -> torch.Tensor:
        if self.imgs[index] is None:
            self.imgs[index] = self._load_image(index)
        return cast(torch.Tensor, self.imgs[index])

    def get_depth(self, index: int) -> torch.Tensor:
        if self.depths[index] is None:
            self.depths[index] = self._load_depth(index)
        return cast(torch.Tensor, self.depths[index])

    def get_mask(self, index: int) -> torch.Tensor:
        if self.masks[index] is None:
            self.masks[index] = self._load_mask(index)
        return cast(torch.Tensor, self.masks[index])

    def _load_image(self, index: int) -> torch.Tensor:
        path = self.data_dir / "images" / f"{self.frame_names[index]}.png"
        if not path.exists():
            raise FileNotFoundError(path)
        image = np.array(Image.open(path))
        if image.ndim == 3 and image.shape[-1] == 4:
            alpha = image[..., 3:4].astype(np.float32) / 255.0
            rgb = image[..., :3].astype(np.float32) * alpha + (1.0 - alpha) * 255.0
        else:
            rgb = image[..., :3].astype(np.float32)
        return _resize_rgb(rgb, self.target_size)

    def _load_mask_bool(self, index: int) -> np.ndarray:
        name = self.frame_names[index]
        image = np.array(Image.open(self.data_dir / "images" / f"{name}.png"))
        mask = np.array(Image.open(self.data_dir / "masks" / f"{name}.png")) > 0
        if image.ndim == 3 and image.shape[-1] == 4:
            mask = mask & (image[..., 3] > 127)
        return mask

    def _load_mask(self, index: int) -> torch.Tensor:
        mask = self._load_mask_bool(index).astype(np.float32)
        mask = _resize_float_image(mask, self.target_size, mode="nearest") > 0.5
        tri_mask = torch.full(mask.shape, -1.0, dtype=torch.float32)
        tri_mask[mask] = 1.0
        return tri_mask

    def _load_depth(self, index: int) -> torch.Tensor:
        path = self.data_dir / "depth" / f"{self.frame_names[index]}.png"
        if not path.exists():
            raise FileNotFoundError(path)
        depth = np.array(Image.open(path)).astype(np.float32) / self.depth_scale
        depth[~self._load_mask_bool(index)] = 0.0
        depth = _resize_float_image(depth, self.target_size, mode="bilinear")
        return depth / self.scene_norm_scale

    def get_tracks_3d(
        self, num_samples: int, start: int = 0, end: int = -1, step: int = 1, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_key = (num_samples, start, end, step)
        if cache_key in self._tracks_cache:
            tracks, visibles, invisibles, confidences, colors, mask_ids = self._tracks_cache[cache_key]
            self.last_track_mask_ids = mask_ids.clone()
            return tracks.clone(), visibles.clone(), invisibles.clone(), confidences.clone(), colors.clone()

        coords, visibs, mask_ids = self._load_track_arrays()
        repeat_factor = self.motion_num_frames // self.track_phase_frames
        coords = coords.repeat((repeat_factor, 1, 1))
        visibs = visibs.repeat((repeat_factor, 1))

        if end < 0:
            end = self.motion_num_frames + 1 + end
        end = min(end, self.motion_num_frames)
        frame_idcs = list(range(start, end, step))
        if not frame_idcs:
            frame_idcs = list(range(0, self.motion_num_frames, step))

        coords = coords[frame_idcs]
        visibs = visibs[frame_idcs]
        tracks_3d = self._normalize_points(coords).permute(1, 0, 2).contiguous()
        visibles = visibs.permute(1, 0).float().contiguous()
        invisibles = 1.0 - visibles
        confidences = visibles.clone()

        keep_mask = visibles.sum(dim=1) > 0
        tracks_3d = tracks_3d[keep_mask]
        visibles = visibles[keep_mask]
        invisibles = invisibles[keep_mask]
        confidences = confidences[keep_mask]
        mask_ids = mask_ids[keep_mask]

        if num_samples > 0 and len(tracks_3d) > num_samples:
            sel_idcs = torch.randperm(len(tracks_3d))[:num_samples]
            tracks_3d = tracks_3d[sel_idcs]
            visibles = visibles[sel_idcs]
            invisibles = invisibles[sel_idcs]
            confidences = confidences[sel_idcs]
            mask_ids = mask_ids[sel_idcs]

        colors = self._sample_track_colors(tracks_3d[:, 0, :], frame_idcs[0])
        self.last_track_mask_ids = mask_ids.clone()
        self._tracks_cache[cache_key] = (
            tracks_3d.clone(),
            visibles.clone(),
            invisibles.clone(),
            confidences.clone(),
            colors.clone(),
            mask_ids.clone(),
        )
        guru.info(
            f"Loaded Articulat3D tracks: xyz={tuple(tracks_3d.shape)}, "
            f"mask_ids={torch.unique(mask_ids).tolist()}"
        )
        return tracks_3d, visibles, invisibles, confidences, colors

    def _sample_track_colors(self, pts_world: torch.Tensor, frame_idx: int) -> torch.Tensor:
        img = self.get_image(frame_idx)
        H, W = img.shape[:2]
        w2c = self.w2cs[frame_idx]
        K = self.Ks[frame_idx]

        pts_cam = (w2c[:3, :3] @ pts_world.T).T + w2c[:3, 3]
        z = pts_cam[:, 2].clamp(min=1e-6)
        u = (K[0, 0] * pts_cam[:, 0] + K[0, 2] * pts_cam[:, 2]) / z
        v = (K[1, 1] * pts_cam[:, 1] + K[1, 2] * pts_cam[:, 2]) / z

        valid = (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1) & (pts_cam[:, 2] > 1e-6)
        colors = torch.zeros((len(pts_world), 3), dtype=img.dtype)
        if valid.any():
            colors[valid] = img[v[valid].round().long(), u[valid].round().long()]
        if (~valid).any():
            colors[~valid] = img.reshape(-1, 3).mean(dim=0)
        return colors

    def get_bkgd_points(self, num_samples: int, **kwargs):
        raise NotImplementedError("Articulat3D training currently uses num_bg=0")

    def __getitem__(self, index: int):
        tri_mask = self.get_mask(index)
        valid_mask = tri_mask != 0
        mask = tri_mask == 1
        return {
            "frame_names": self.frame_names[index],
            "ts": torch.tensor(self._ts[index], dtype=torch.long),
            "w2cs": self.w2cs[index],
            "Ks": self.Ks[index],
            "imgs": self.get_image(index),
            "depths": self.get_depth(index),
            "masks": mask.float(),
            "valid_masks": valid_mask.float(),
        }


class Articulat3DVideoView(Dataset):
    def __init__(self, dataset: Articulat3DDataset, include_static: bool = False):
        super().__init__()
        self.dataset = dataset
        self.include_static = include_static
        self.fps = 15.0
        self.length = len(dataset) if include_static else dataset.motion_num_frames

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        tri_mask = self.dataset.get_mask(index)
        return {
            "frame_names": self.dataset.frame_names[index],
            "ts": torch.tensor(self.dataset._ts[index], dtype=torch.long),
            "w2cs": self.dataset.w2cs[index],
            "Ks": self.dataset.Ks[index],
            "imgs": self.dataset.get_image(index),
            "depths": self.dataset.get_depth(index),
            "masks": (tri_mask == 1).float(),
        }
