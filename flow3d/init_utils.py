import time
from typing import Literal

import cupy as cp
import imageio.v3 as iio
import numpy as np

# from pytorch3d.ops import sample_farthest_points
import roma
import torch
import torch.nn.functional as F
from torch import Tensor
from cuml import HDBSCAN, KMeans
from loguru import logger as guru
from matplotlib.pyplot import get_cmap
from tqdm import tqdm
from viser import ViserServer

from flow3d.loss_utils import (
    compute_accel_loss,
    compute_se3_smoothness_loss,
    compute_z_acc_loss,
    get_weights_for_procrustes,
    knn,
    masked_l1_loss,
)
from flow3d.params import GaussianParams, MotionBases, CameraPoses
from flow3d.tensor_dataclass import StaticObservations, TrackObservations
from flow3d.transforms import cont_6d_to_rmat, rt_to_mat4, solve_procrustes
from flow3d.vis.utils import draw_keypoints_video, get_server, project_2d_tracks


def init_trainable_poses(w2cs: Tensor) -> CameraPoses:
    N, _, _ = w2cs.shape
    Rs = w2cs[:, :3, :3]
    ts = w2cs[:, :3, 3:]

    return CameraPoses(Rs, ts)

def init_fg_from_tracks_3d(
    cano_t: int, tracks_3d: TrackObservations, motion_coefs: torch.Tensor
) -> GaussianParams:
    """
    using dataclasses individual tensors so we know they're consistent
    and are always masked/filtered together
    # 这段代码将 3D 点云正式转化为高斯球
    """
    num_fg = tracks_3d.xyz.shape[0]

    # 1. 颜色初始化，直接使用观测到的点云颜色
    # Initialize gaussian colors.
    colors = torch.logit(tracks_3d.colors)
    
    # 2. 尺度初始化
    # Initialize gaussian scales: find the average of the three nearest
    # neighbors in the first frame for each point and use that as the
    # scale.
    ## 对于每个点，找到它在标准时刻 cano_t 离他最近的 3 个邻居的距离，然后取平均值作为尺度
    ## 因为如果和邻居的距离过大，这个尺度就需要大一点来弥补空隙
    dists, _ = knn(tracks_3d.xyz[:, cano_t], 3)
    dists = torch.from_numpy(dists)
    scales = dists.mean(dim=-1, keepdim=True)
    scales = scales.clamp(torch.quantile(scales, 0.05), torch.quantile(scales, 0.95))
    scales = torch.log(scales.repeat(1, 3))
    # 3. 位置初始化
    # Initialize gaussian means.
    means = tracks_3d.xyz[:, cano_t]
    # 4. 方向初始化，也就是旋转初始化
    # Initialize gaussian orientations as random.
    quats = torch.rand(num_fg, 4)
    # 5. 不透明度初始化
    # Initialize gaussian opacities.
    opacities = torch.logit(torch.full((num_fg,), 0.7))
    # 6. 打包返回
    gaussians = GaussianParams(means, quats, scales, colors, opacities, motion_coefs)
    return gaussians


def init_bg(
    points: StaticObservations,
) -> GaussianParams:
    """
    using dataclasses instead of individual tensors so we know they're consistent
    and are always masked/filtered together
    """
    num_init_bg_gaussians = points.xyz.shape[0]
    bg_scene_center = points.xyz.mean(0)
    bg_points_centered = points.xyz - bg_scene_center
    bg_min_scale = bg_points_centered.quantile(0.05, dim=0)
    bg_max_scale = bg_points_centered.quantile(0.95, dim=0)
    bg_scene_scale = torch.max(bg_max_scale - bg_min_scale).item() / 2.0
    bkdg_colors = torch.logit(points.colors)

    # Initialize gaussian scales: find the average of the three nearest
    # neighbors in the first frame for each point and use that as the
    # scale.
    dists, _ = knn(points.xyz, 3)
    dists = torch.from_numpy(dists)
    bg_scales = dists.mean(dim=-1, keepdim=True)
    bkdg_scales = torch.log(bg_scales.repeat(1, 3))

    bg_means = points.xyz

    # Initialize gaussian orientations by normals.
    local_normals = points.normals.new_tensor([[0.0, 0.0, 1.0]]).expand_as(
        points.normals
    )
    bg_quats = roma.rotvec_to_unitquat(
        F.normalize(local_normals.cross(points.normals), dim=-1)
        * (local_normals * points.normals).sum(-1, keepdim=True).acos_()
    ).roll(1, dims=-1)
    bg_opacities = torch.logit(torch.full((num_init_bg_gaussians,), 0.7))
    gaussians = GaussianParams(
        bg_means,
        bg_quats,
        bkdg_scales,
        bkdg_colors,
        bg_opacities,
        scene_center=bg_scene_center,
        scene_scale=bg_scene_scale,
    )
    return gaussians


def init_motion_params_with_procrustes(
    tracks_3d: TrackObservations,
    num_bases: int,
    rot_type: Literal["quat", "6d"],
    cano_t: int,
    cluster_init_method: str = "kmeans",
    min_mean_weight: float = 0.1,
    vis: bool = False,
    port: int | None = None,
) -> tuple[MotionBases, torch.Tensor, TrackObservations]:
    r'''
    从一堆杂乱无章的 3D 轨迹中，悟出场景里有几个物体在动，以及他们是怎么动的
    输入是 tracks_3d 成千上万的 3D 点轨迹，输出是几个简洁的“运动基”
    tracks_3d:
        xyz: (N, T, 3)
        visibles: (N, T)
        invisibles: (N, T)
        confidences: (N, T)
        colors: (N, 3)
    '''
    device = tracks_3d.xyz.device
    num_frames = tracks_3d.xyz.shape[1] 
    # sample centers and get initial se3 motion bases by solving procrustes

    # 1. 准备工作和数据清洗
    # 提取标准帧的 3D 点坐标，并计算每个维度的中位数，得到场景中心
    means_cano = tracks_3d.xyz[:, cano_t].clone()  # [num_gaussians, 3] [N, 1, 3]
    scene_center = means_cano.median(dim=0).values
    print(f"标准时刻场景中心为：{scene_center}")
    # 计算每个 3D 点坐标到场景中心的距离
    dists = torch.norm(means_cano - scene_center, dim=-1)
    # 计算距离的 95% 分位数，作为距离阈值
    dists_th = torch.quantile(dists, 0.95)
    # 根据距离阈值，筛选出有效的 3D 点
    valid_mask = dists < dists_th
    # remove tracks that are not visible in any frame
    # 如果一个点在整个视频里都没有出现过，就扔掉
    valid_mask = valid_mask & tracks_3d.visibles.any(dim=1)
    print(f"有效的 3D 点数量为：{valid_mask.sum()}")
    # 只保留有效的 3D 点
    tracks_3d = tracks_3d.filter_valid(valid_mask)
    if vis and port is not None:
        server = get_server(port)
        try:
            pts = tracks_3d.xyz.cpu().numpy()
            clrs = tracks_3d.colors.cpu().numpy()
            while True:
                for t in range(num_frames):
                    server.scene.add_point_cloud("points", pts[:, t], clrs)
                    time.sleep(0.3)
        except KeyboardInterrupt:
            pass
    means_cano = means_cano[valid_mask]

    # 2. 运动聚类，使用 K-means 算法
    ## 输入所有 3D 点的轨迹 cluster_init_method=kmeans
    ## 最后得到的 labels（每个点属于哪个类），sampled_centers（每个类的中心点坐标），num_bases（聚类数量，也就是运动基数量）
    sampled_centers, num_bases, labels = sample_initial_bases_centers(
        cluster_init_method, cano_t, tracks_3d, num_bases
    )
    ids, counts = labels.unique(return_counts=True)
    # 原本 TAPIP3D 的结果就是 71 个点，然后现在过滤后查询轨迹点只有 67 个了
    # 将下面两行注释掉了
    # ids =ids[counts > 100]
    # num_bases = len(ids)
    ################################################################################################################
    ########################## 避免因为 TAPIP3D 轨迹数量较少而把所有聚类都丢掉
    total_points = counts.sum().item()
    min_cluster_points = max(10, int(0.01 * total_points))
    valid_mask = counts >= min_cluster_points
    if valid_mask.any():
        ids = ids[valid_mask]
        counts = counts[valid_mask]
    else:
        topk = min(len(ids), num_bases,  max(1, len(ids)))
        top_vals, top_idx = torch.topk(counts, k=topk)
        ids = ids[top_idx]
        counts = top_vals
    if len(ids) == 0:
        raise RuntimeError(
            "No valid clusters found for motion bases. "
            "Try reducing num_motion_bases or ensuring tracks_3d contains at least a few hundred points."
        )
    num_bases = min(len(ids), num_bases)
    ids = ids[:num_bases]
    ################################################################################################################
    sampled_centers = sampled_centers[:, ids]
    print(f"运动基数量为：{num_bases}，运动基的中心点坐标为：{sampled_centers}")

    # 3. 计算运动系数，权重和距离成反比
    # compute basis weights from the distance to the cluster centers
    ## 计算标准帧的每个点到每个运动基的中心点的距离
    dists2centers = torch.norm(means_cano[:, None] - sampled_centers, dim=-1)
    ## 使用高斯核函数将距离转化为权重，距离越近，权重越大 [N, K]，每个点有 K 个权重
    motion_coefs = 10 * torch.exp(-dists2centers)   # 后续没有被用来计算，而是直接返回了

    # 4. 计算运动轨迹
    init_rots, init_ts = [], []
    if rot_type == "quat": 
        id_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        rot_dim = 4
    else:   # 走这个分支 6D
        id_rot = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device)
        rot_dim = 6
    # 初始化运动基的旋转和平移，初始化为单位旋转和平移
    init_rots = id_rot.reshape(1, 1, rot_dim).repeat(num_bases, num_frames, 1)  # [K, T, 6] # 每一帧的每个簇的双四元数
    init_ts = torch.zeros(num_bases, num_frames, 3, device=device)  # [K, T, 3] # 每一帧的每个簇的平移向量  
    errs_before = np.full((num_bases, num_frames), -1.0)
    errs_after = np.full((num_bases, num_frames), -1.0)

    tgt_ts = list(range(cano_t - 1, -1, -1)) + list(range(cano_t, num_frames))
    # print(f"目标时间步列表为：{tgt_ts}")    # 其实就是从标准帧到目标帧的目标列表，因为要计算的就是从标准帧到目标帧的运动
    skipped_ts = {}
    # 遍历每个运动基 ids 其实就是 0 1 2 ，如果 num_bases=3 的话
    for n, cluster_id in enumerate(ids):
        # 找到属于当前运动基的点
        mask_in_cluster = labels == cluster_id  # [67] 的 True or False
        # 获取当前运动基的点轨迹
        cluster = tracks_3d.xyz[mask_in_cluster].transpose(
            0, 1
        )  # [num_frames, n_pts, 3] 将 N 和 T 调换位置了 [T, p_label_i, 3]
        visibilities = tracks_3d.visibles[mask_in_cluster].swapaxes(
            0, 1
        )  # [num_frames, n_pts]
        confidences = tracks_3d.confidences[mask_in_cluster].swapaxes(
            0, 1
        )  # [num_frames, n_pts]
        # 得到这个聚类簇后，计算每个点到这个簇的中心点的距离，然后变成权重
        weights = get_weights_for_procrustes(cluster, visibilities) # [T, p_label_i]
        prev_t = cano_t
        cluster_skip_ts = []
        for cur_t in tgt_ts:    # 对于每一个目标帧
            # compute pairwise transform from cano_t
            ## 中心距离衰减+可见性 权重
            procrustes_weights = (weights[cano_t] * weights[cur_t] * (confidences[cano_t] + confidences[cur_t]) / 2)
            # 如果有效权重总和太小（可见点太少或置信度太低），认为该帧数据不足以稳健估计刚体变换。
            if procrustes_weights.sum() < min_mean_weight * num_frames:
                init_rots[n, cur_t] = init_rots[n, prev_t]
                init_ts[n, cur_t] = init_ts[n, prev_t]
                cluster_skip_ts.append(cur_t)
            else:
                # 已知第 0 帧的一堆点 P0 和第 t 帧的同一堆点 Pt，求一个最佳的旋转 R 和平移 t，使得 R x P0 + t = Pt。
                se3, (err, err_before) = solve_procrustes(
                    src=cluster[cano_t],
                    dst=cluster[cur_t],
                    weights=procrustes_weights,
                    enforce_se3=True,
                    rot_type=rot_type
                )
                init_rot, init_t, _ = se3
                assert init_rot.shape[-1] == rot_dim
                # double cover
                if rot_type == "quat" and torch.linalg.norm(
                    init_rot - init_rots[n][prev_t]
                ) > torch.linalg.norm(-init_rot - init_rots[n][prev_t]):
                    init_rot = -init_rot
                init_rots[n, cur_t] = init_rot
                init_ts[n, cur_t] = init_t
                if np.isnan(err):
                    print(f"第 {cur_t} 帧的误差为：{err}")
                    print(f"第 {cur_t} 帧的权重为：{procrustes_weights.isnan().sum()}")
                if np.isnan(err_before):
                    print(f"第 {cur_t} 帧的优化前误差为：{err_before}")
                    print(f"第 {cur_t} 帧的权重为：{procrustes_weights.isnan().sum()}")
                errs_after[n, cur_t] = err
                errs_before[n, cur_t] = err_before
            prev_t = cur_t
        # 记录该簇/基在初始化时因权重不足而跳过、继承前一帧姿态的帧列表。
        skipped_ts[cluster_id.item()] = cluster_skip_ts
    guru.info(f"继承前一帧的姿态的帧列表为：{skipped_ts}")
    # 打印 Procrustes 初始化的误差统计：
    # errs_before: 优化前的误差
    # errs_after: 优化后（即求解 Procrustes 后）的误差
    # 这里只统计了误差大于 0 的有效值，展示中位数和平均值
    # guru.info(
    #     "procrustes init median error: {:.5f} => {:.5f}".format(
    #         np.median(errs_before[errs_before > 0]),
    #         np.median(errs_after[errs_after > 0]),
    #     )
    # )
    # guru.info(
    #     "procrustes init mean error: {:.5f} => {:.5f}".format(
    #         np.mean(errs_before[errs_before > 0]), np.mean(errs_after[errs_after > 0])
    #     )
    # )
    # 打印最终生成的张量形状：
    # init_rots: 初始旋转矩阵 [num_bases, num_frames, rot_dim]
    # init_ts: 初始平移向量 [num_bases, num_frames, 3]
    # motion_coefs: 运动系数（蒙皮权重） [num_points, num_bases]
    guru.info(f"初始旋转矩阵：{init_rots.shape}, 初始平移向量：{init_ts.shape}, 运动系数：{motion_coefs.shape}")

    if vis:
        server = get_server(port)
        # 找到距离每个聚类中心最近的点的索引
        center_idcs = torch.argmin(dists2centers, dim=0)
        print(f"{dists2centers.shape=} {center_idcs.shape=}")
        # 可视化初始化的 SE3 运动轨迹
        vis_se3_init_3d(server, init_rots, init_ts, means_cano[center_idcs])
        # 可视化这些中心点的 3D 轨迹
        vis_tracks_3d(server, tracks_3d.xyz[center_idcs].numpy(), name="center_tracks")
        import ipdb

        ipdb.set_trace()
    # bases 运动的骨架，每个运动基的旋转和平移
    # motion_coefs 蒙皮权重，每个 3D 点分别受这 3 个刚体多大的影响
    # tracks_3d：清洗后的原始观测数据，用于后续的 loss 计算
    bases = MotionBases(init_rots, init_ts)
    '''
    bases [num_bases, num_frames, 4|6] [K, T, 4|6]
    motion_coefs [num_points, num_bases] [N, K]
    tracks_3d 经过清洗、筛选后的原始观测数据 xyz [N, T, 3] visibles[N, T] confidences[N, T] colors[N, 3]
    其实就是将离群点丢掉以及无效点（全程不可见的点）剔除了。
    '''
    return bases, motion_coefs, tracks_3d

def sample_initial_bases_centers(
    mode: str, cano_t: int, tracks_3d: TrackObservations, num_bases: int
):
    """
    :param mode: "farthest" | "hdbscan" | "kmeans"
    :param tracks_3d: 
                # tracks_3d (x, y, z): (N, T, 3) 4582 10 3，中间这个 T 是关键帧个数，由于传进来的 step 参数是 num_frames // 10，所以 T = 10
                # visibles: (N, T)  4582 10
                # invisibles: (N, T) 4582 10
                # confidences: (N, T) 4582 10
                # colors: (N, 3) 4582 3
    :param cano_t: canonical index
    :param num_bases: number of SE3 bases
    
    根据 3D 轨迹的运动方向特征对所有轨迹做聚类，输出每个簇的中心点，作为初始运动基的中心
    """
    assert mode in ["farthest", "hdbscan", "kmeans"]
    means_canonical = tracks_3d.xyz[:, cano_t].clone()
    # if mode == "farthest":
    #     vis_mask = tracks_3d.visibles[:, cano_t]
    #     sampled_centers, _ = sample_farthest_points(
    #         means_canonical[vis_mask][None],
    #         K=num_bases,
    #         random_start_point=True,
    #     )  # [1, num_bases, 3]
    #     dists2centers = torch.norm(means_canonical[:, None] - sampled_centers, dim=-1).T
    #     return sampled_centers, num_bases, dists2centers

    # linearly interpolate missing 3d points
    xyz = cp.asarray(tracks_3d.xyz)
    print(f"3D 轨迹点的形状为：tracks_3d.xyz.shape = {xyz.shape}")
    visibles = cp.asarray(tracks_3d.visibles)

    num_tracks = xyz.shape[0]
    xyz_interp = batched_interp_masked(xyz, visibles)

    # num_vis = 50
    # server = get_server(port=8890)
    # idcs = np.random.choice(num_tracks, num_vis)
    # labels = np.linspace(0, 1, num_vis)
    # vis_tracks_3d(server, tracks_3d.xyz[idcs].get(), labels, name="raw_tracks")
    # vis_tracks_3d(server, xyz_interp[idcs].get(), labels, name="interp_tracks")
    # import ipdb; ipdb.set_trace()
    
    # 提取运动方向的特征
    # 计算两两相邻帧之间的运动方向
    velocities = xyz_interp[:, 1:] - xyz_interp[:, :-1]
    # 归一化方向
    vel_dirs = (
        velocities / (cp.linalg.norm(velocities, axis=-1, keepdims=True) + 1e-5)
    ).reshape((num_tracks, -1))

    # [num_bases, num_gaussians]
    if mode == "kmeans":
        model = KMeans(n_clusters=num_bases)
    else:
        model = HDBSCAN(min_cluster_size=20, max_cluster_size=num_tracks // 4)
    # 聚类
    model.fit(vel_dirs)
    labels = model.labels_  # 得到类别标签
    num_bases = labels.max().item() + 1
    print(f"最终聚类后的类别数量：{num_bases}")
    # Ensure labels are a torch tensor on the same device as means_canonical
    labels_torch = torch.from_numpy(cp.asnumpy(labels)).to(means_canonical.device)
    # 对每个簇 i 在标准帧上取该簇的所有 3D 点坐标的中位数，作为该簇的中心点
    sampled_centers = torch.stack(
        [
            means_canonical[labels_torch == i].median(dim=0).values
            for i in range(num_bases)
        ]
    )[None]
    # print("number of {} clusters: ".format(mode), num_bases)
    
    return sampled_centers, num_bases, labels_torch  # 返回每个簇的中心点，以及类别标签

def run_initial_optim(
    fg: GaussianParams,
    bases: MotionBases,
    tracks_3d: TrackObservations,
    Ks: torch.Tensor,
    w2cs: torch.Tensor,
    num_iters: int = 1000,
    use_depth_range_loss: bool = False,
):
    """
    该函数负责对初始化后的 3D 模型进行第一轮粗修
    :param motion_rots: [num_bases, num_frames, 4|6]
    :param motion_transls: [num_bases, num_frames, 3]
    :param motion_coefs: [num_bases, num_frames]
    :param means: [num_gaussians, 3]
    """
    optimizer = torch.optim.Adam(
        [
            {"params": bases.params["rots"], "lr": 1e-2},   # 运动基的旋转
            {"params": bases.params["transls"], "lr": 3e-2},   # 运动基的平移
            {"params": fg.params["motion_coefs"], "lr": 1e-2},   # 蒙皮权重
            {"params": fg.params["means"], "lr": 1e-3},   # 高斯球的中心
        ],
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.1 ** (1 / num_iters)
    )
    G = fg.params.means.shape[0]
    num_frames = bases.num_frames
    device = bases.params["rots"].device

    w_smooth_func = lambda i, min_v, max_v, th: (
        min_v if i <= th else (max_v - min_v) * (i - th) / (num_iters - th) + min_v
    )

    gt_2d, gt_depth = project_2d_tracks(
        tracks_3d.xyz.swapaxes(0, 1), Ks, w2cs, return_depth=True
    )
    # (G, T, 2)
    gt_2d = gt_2d.swapaxes(0, 1)
    # (G, T)
    gt_depth = gt_depth.swapaxes(0, 1)

    ts = torch.arange(0, num_frames, device=device)
    ts_clamped = torch.clamp(ts, min=1, max=num_frames - 2)
    ts_neighbors = torch.cat((ts_clamped - 1, ts_clamped, ts_clamped + 1))  # i (3B,)

    pbar = tqdm(range(0, num_iters))
    for i in pbar:
        coefs = fg.get_coefs()
        transfms = bases.compute_transforms(ts, coefs)
        positions = torch.einsum(
            "pnij,pj->pni",
            transfms,
            F.pad(fg.params["means"], (0, 1), value=1.0),
        )

        loss = 0.0
        # 3D 轨迹误差，让模型预测每个点在每一帧的 3D 坐标
        track_3d_loss = masked_l1_loss(
            positions,
            tracks_3d.xyz,
            (tracks_3d.visibles.float() * tracks_3d.confidences)[..., None],
        )
        loss += track_3d_loss * 1.0

        # loss_2d 2D 投影误差
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
                loss_depth_in_range += (near_depths - pred_depth)[
                    pred_depth < near_depths
                ].mean()
            if (pred_depth > far_depths).any():
                loss_depth_in_range += (pred_depth - far_depths)[
                    pred_depth > far_depths
                ].mean()

            loss += loss_depth_in_range * w_smooth_func(i, 0.05, 0.5, 400)

        # 蒙皮权重稀疏损失，让蒙皮权重尽可能稀疏
        motion_coef_sparse_loss = 1 - (coefs**2).sum(dim=-1).mean()
        loss += motion_coef_sparse_loss * 0.01

        # motion basis should be smooth.
        w_smooth = w_smooth_func(i, 0.01, 0.1, 400)
        small_acc_loss = compute_se3_smoothness_loss(
            bases.params["rots"], bases.params["transls"]
        )
        loss += small_acc_loss * w_smooth

        small_acc_loss_tracks = compute_accel_loss(positions)
        loss += small_acc_loss_tracks * w_smooth * 0.5

        transfms_nbs = bases.compute_transforms(ts_neighbors, coefs)
        means_nbs = torch.einsum(
            "pnij,pj->pni", transfms_nbs, F.pad(fg.params["means"], (0, 1), value=1.0)
        )  # (G, 3n, 3)
        means_nbs = means_nbs.reshape(means_nbs.shape[0], 3, -1, 3)  # [G, 3, n, 3]
        z_accel_loss = compute_z_acc_loss(means_nbs, w2cs)
        loss += z_accel_loss * 0.1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        pbar.set_description(
            f"总损失：{loss.item():.3f} "
            f"3D 轨迹误差：{track_3d_loss.item():.3f} "
            f"蒙皮权重稀疏损失：{motion_coef_sparse_loss.item():.3f} "
            f"运动基平滑损失：{small_acc_loss.item():.3f} "
            f"3D 轨迹加速度损失：{small_acc_loss_tracks.item():.3f} "
            f"z 方向深度一致性损失：{z_accel_loss.item():.3f} "
        )


def random_quats(N: int) -> torch.Tensor:
    u = torch.rand(N, 1)
    v = torch.rand(N, 1)
    w = torch.rand(N, 1)
    quats = torch.cat(
        [
            torch.sqrt(1.0 - u) * torch.sin(2.0 * np.pi * v),
            torch.sqrt(1.0 - u) * torch.cos(2.0 * np.pi * v),
            torch.sqrt(u) * torch.sin(2.0 * np.pi * w),
            torch.sqrt(u) * torch.cos(2.0 * np.pi * w),
        ],
        -1,
    )
    return quats


def compute_means(ts, fg: GaussianParams, bases: MotionBases):
    transfms = bases.compute_transforms(ts, fg.get_coefs())
    means = torch.einsum(
        "pnij,pj->pni",
        transfms,
        F.pad(fg.params["means"], (0, 1), value=1.0),
    )
    return means


def vis_init_params(
    server,
    fg: GaussianParams,
    bases: MotionBases,
    name="init_params",
    num_vis: int = 100,
):
    idcs = np.random.choice(fg.num_gaussians, num_vis)
    labels = np.linspace(0, 1, num_vis)
    ts = torch.arange(bases.num_frames, device=bases.params["rots"].device)
    with torch.no_grad():
        pred_means = compute_means(ts, fg, bases)
        vis_means = pred_means[idcs].detach().cpu().numpy()
    vis_tracks_3d(server, vis_means, labels, name=name)


@torch.no_grad()
def vis_se3_init_3d(server, init_rots, init_ts, basis_centers):
    """
    :param init_rots: [num_bases, num_frames, 4|6]
    :param init_ts: [num_bases, num_frames, 3]
    :param basis_centers: [num_bases, 3]
    """
    # visualize the initial centers across time
    rot_dim = init_rots.shape[-1]
    assert rot_dim in [4, 6]
    num_bases = init_rots.shape[0]
    assert init_ts.shape[0] == num_bases
    assert basis_centers.shape[0] == num_bases
    labels = np.linspace(0, 1, num_bases)
    if rot_dim == 4:
        quats = F.normalize(init_rots, dim=-1, p=2)
        rmats = roma.unitquat_to_rotmat(quats.roll(-1, dims=-1))
    else:
        rmats = cont_6d_to_rmat(init_rots)
    transls = init_ts
    transfms = rt_to_mat4(rmats, transls)
    center_tracks3d = torch.einsum(
        "bnij,bj->bni", transfms, F.pad(basis_centers, (0, 1), value=1.0)
    )[..., :3]
    vis_tracks_3d(server, center_tracks3d.cpu().numpy(), labels, name="se3_centers")


@torch.no_grad()
def vis_tracks_2d_video(
    path,
    imgs: np.ndarray,
    tracks_3d: np.ndarray,
    Ks: np.ndarray,
    w2cs: np.ndarray,
    occs=None,
    radius: int = 3,
):
    num_tracks = tracks_3d.shape[0]
    labels = np.linspace(0, 1, num_tracks)
    cmap = get_cmap("gist_rainbow")
    colors = cmap(labels)[:, :3]
    tracks_2d = (
        project_2d_tracks(tracks_3d.swapaxes(0, 1), Ks, w2cs).cpu().numpy()  # type: ignore
    )
    frames = np.asarray(
        draw_keypoints_video(imgs, tracks_2d, colors, occs, radius=radius)
    )
    iio.imwrite(path, frames, fps=15)


def vis_tracks_3d(
    server: ViserServer,
    vis_tracks: np.ndarray,
    vis_label: np.ndarray | None = None,
    name: str = "tracks",
):
    """
    :param vis_tracks (np.ndarray): (N, T, 3)
    :param vis_label (np.ndarray): (N)
    """
    cmap = get_cmap("gist_rainbow")
    if vis_label is None:
        vis_label = np.linspace(0, 1, len(vis_tracks))
    colors = cmap(np.asarray(vis_label))[:, :3]
    guru.info(f"{colors.shape=}, {vis_tracks.shape=}")
    N, T = vis_tracks.shape[:2]
    vis_tracks = np.asarray(vis_tracks)
    for i in range(N):
        server.scene.add_spline_catmull_rom(
            f"/{name}/{i}/spline", vis_tracks[i], color=colors[i], segments=T - 1
        )
        server.scene.add_point_cloud(
            f"/{name}/{i}/start",
            vis_tracks[i, [0]],
            colors=colors[i : i + 1],
            point_size=0.05,
            point_shape="circle",
        )
        server.scene.add_point_cloud(
            f"/{name}/{i}/end",
            vis_tracks[i, [-1]],
            colors=colors[i : i + 1],
            point_size=0.05,
            point_shape="diamond",
        )


def interp_masked(vals: cp.ndarray, mask: cp.ndarray, pad: int = 1) -> cp.ndarray:
    """
    hacky way to interpolate batched with cupy
    by concatenating the batches and pad with dummy values
    :param vals: [B, M, *]
    :param mask: [B, M]
    """
    assert mask.ndim == 2
    assert vals.shape[:2] == mask.shape

    B, M = mask.shape

    # get the first and last valid values for each track
    sh = vals.shape[2:]
    vals = vals.reshape((B, M, -1))
    D = vals.shape[-1]
    first_val_idcs = cp.argmax(mask, axis=-1)
    last_val_idcs = M - 1 - cp.argmax(cp.flip(mask, axis=-1), axis=-1)
    bidcs = cp.arange(B)

    v0 = vals[bidcs, first_val_idcs][:, None]
    v1 = vals[bidcs, last_val_idcs][:, None]
    m0 = mask[bidcs, first_val_idcs][:, None]
    m1 = mask[bidcs, last_val_idcs][:, None]
    if pad > 1:
        v0 = cp.tile(v0, [1, pad, 1])
        v1 = cp.tile(v1, [1, pad, 1])
        m0 = cp.tile(m0, [1, pad])
        m1 = cp.tile(m1, [1, pad])

    vals_pad = cp.concatenate([v0, vals, v1], axis=1)
    mask_pad = cp.concatenate([m0, mask, m1], axis=1)

    M_pad = vals_pad.shape[1]
    vals_flat = vals_pad.reshape((B * M_pad, -1))
    mask_flat = mask_pad.reshape((B * M_pad,))
    idcs = cp.where(mask_flat)[0]

    cx = cp.arange(B * M_pad)
    out = cp.zeros((B * M_pad, D), dtype=vals_flat.dtype)
    
    # Check if there are any visible points to interpolate
    ############################## 增加了下面这个 if ， for 这个语句是原本就有的。
    if len(idcs) > 0:
        for d in range(D):
            out[:, d] = cp.interp(cx, idcs, vals_flat[idcs, d])
    # If no visible points, out remains zeros (already initialized)

    out = out.reshape((B, M_pad, *sh))[:, pad:-pad]
    return out


def batched_interp_masked(
    vals: cp.ndarray, mask: cp.ndarray, batch_num: int = 4096, batch_time: int = 64
):
    assert mask.ndim == 2
    B, M = mask.shape
    out = cp.zeros_like(vals)
    for b in tqdm(range(0, B, batch_num), leave=False):
        for m in tqdm(range(0, M, batch_time), leave=False):
            x = interp_masked(
                vals[b : b + batch_num, m : m + batch_time],
                mask[b : b + batch_num, m : m + batch_time],
            )  # (batch_num, batch_time, *)
            out[b : b + batch_num, m : m + batch_time] = x
    return out
