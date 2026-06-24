import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, cv2
import matplotlib.pyplot as plt
import math
from torch import Tensor

def normalized_quat_to_rotmat(quat: Tensor) -> Tensor:
    """Convert normalized quaternion to rotation matrix.

    Args:
        quat: Normalized quaternion in wxyz convension. (..., 4)

    Returns:
        Rotation matrix (..., 3, 3)
    """
    assert quat.shape[-1] == 4, quat.shape
    w, x, y, z = torch.unbind(quat, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return mat.reshape(quat.shape[:-1] + (3, 3))

# ref: https://github.com/hbb1/2d-gaussian-splatting/blob/61c7b417393d5e0c58b742ad5e2e5f9e9f240cc6/utils/point_utils.py#L26
def _depths_to_points(depthmap, world_view_transform, full_proj_transform, fx, fy):
    c2w = (world_view_transform.T).inverse()
    H, W = depthmap.shape[:2]
    intrins = torch.tensor(
        [[fx, 0., W/2.],
        [0., fy, H/2.],
        [0., 0., 1.0]]
    ).float().cuda()

    import pdb
    # pdb.set_trace()

    grid_x, grid_y = torch.meshgrid(
        torch.arange(W, device="cuda").float(),
        torch.arange(H, device="cuda").float(),
        indexing="xy",
    )
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(
        -1, 3
    )
    rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points


def _depth_to_normal(depth, world_view_transform, full_proj_transform, fx, fy):
    points = _depths_to_points(
        depth, world_view_transform, full_proj_transform, fx, fy,
    ).reshape(*depth.shape[:2], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output


def depth_to_normal(depths, camtoworlds, Ks, near_plane, far_plane):
    import pdb
    # pdb.set_trace()
    height, width = depths.shape[1:3]
    viewmats = torch.linalg.inv(camtoworlds)  # [C, 4, 4]

    normals = []
    for cid, depth in enumerate(depths):
        fx = Ks[cid, 0, 0].item()
        fy = Ks[cid, 1, 1].item()
        
        # 打印调试信息
        # print(f"\n[DEBUG] depth_to_normal - frame {cid}:")
        # print(f"  image size: width={width}, height={height}")
        # print(f"  camera intrinsics: fx={fx}, fy={fy}")
        # print(f"  depth range: near_plane={near_plane}, far_plane={far_plane}")
        
        # 安全检查：防止 fx 或 fy 为 0 或异常大
        if fx <= 0 or fy <= 0 or not math.isfinite(fx) or not math.isfinite(fy):
            import warnings
            warnings.warn(f"Invalid camera intrinsics: fx={fx}, fy={fy}. Using default values.")
            fx = max(fx, width / 2.0) if fx <= 0 else fx
            fy = max(fy, height / 2.0) if fy <= 0 else fy
        
        FoVx = 2 * math.atan(width / (2 * fx))
        FoVy = 2 * math.atan(height / (2 * fy))
        
        # print(f"  FoVx={FoVx:.8f} rad ({math.degrees(FoVx):.2f} deg), FoVy={FoVy:.8f} rad ({math.degrees(FoVy):.2f} deg)")
        
        # 安全检查：防止 FoV 为 0 或异常小
        min_fov = 1e-6  # 最小视场角（弧度）
        if FoVx < min_fov:
            # print(f"  WARNING: FoVx too small ({FoVx}), clamping to {min_fov}")
            FoVx = min_fov
        if FoVy < min_fov:
            # print(f"  WARNING: FoVy too small ({FoVy}), clamping to {min_fov}")
            FoVy = min_fov
        
        world_view_transform = viewmats[cid].transpose(0, 1)
        projection_matrix = _getProjectionMatrix(
            znear=near_plane, zfar=far_plane, fovX=FoVx, fovY=FoVy, device=depths.device
        ).transpose(0, 1)
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
        ).squeeze(0)
        normal = _depth_to_normal(depth, world_view_transform, full_proj_transform, fx, fy)
        normals.append(normal)
    normals = torch.stack(normals, dim=0)
    return normals


def _getProjectionMatrix(znear, zfar, fovX, fovY, device="cuda"):
    # 打印调试信息
    # print(f"\n[DEBUG] _getProjectionMatrix:")
    # print(f"  Input: znear={znear}, zfar={zfar}, fovX={fovX:.8f} rad ({math.degrees(fovX):.2f} deg), fovY={fovY:.8f} rad ({math.degrees(fovY):.2f} deg)")
    
    # 安全检查：确保 znear 和 zfar 有效
    if znear <= 0 or not math.isfinite(znear):
        import warnings
        warnings.warn(f"Invalid znear ({znear}), using default value 0.1")
        znear = 0.1
    if zfar <= znear or not math.isfinite(zfar):
        import warnings
        warnings.warn(f"Invalid zfar ({zfar}), adjusting to znear + 1.0")
        zfar = znear + 1.0
    
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))
    
    # print(f"  tanHalfFovX={tanHalfFovX:.8f}, tanHalfFovY={tanHalfFovY:.8f}")

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    
    # print(f"  top={top:.8f}, bottom={bottom:.8f}, right={right:.8f}, left={left:.8f}")

    P = torch.zeros(4, 4, device=device)

    z_sign = 1.0

    # 安全检查：防止除零错误
    right_minus_left = right - left
    top_minus_bottom = top - bottom
    zfar_minus_znear = zfar - znear
    
    # print(f"  right - left={right_minus_left:.10f}, top - bottom={top_minus_bottom:.10f}, zfar - znear={zfar_minus_znear:.10f}")
    
    if abs(right_minus_left) < 1e-6:
        import warnings
        warnings.warn(f"right - left is too small ({right_minus_left}), znear={znear}, fovX={fovX}, using default value")
        right_minus_left_old = right_minus_left
        right_minus_left = max(2.0 * tanHalfFovX * znear, 1e-3)
        if abs(right_minus_left) < 1e-6:
            right_minus_left = 1e-3  # 最后的保险
        # print(f"  WARNING: right - left adjusted from {right_minus_left_old:.10f} to {right_minus_left:.10f}")
    if abs(top_minus_bottom) < 1e-6:
        import warnings
        warnings.warn(f"top - bottom is too small ({top_minus_bottom}), znear={znear}, fovY={fovY}, using default value")
        top_minus_bottom_old = top_minus_bottom
        top_minus_bottom = max(2.0 * tanHalfFovY * znear, 1e-3)
        if abs(top_minus_bottom) < 1e-6:
            top_minus_bottom = 1e-3  # 最后的保险
        # print(f"  WARNING: top - bottom adjusted from {top_minus_bottom_old:.10f} to {top_minus_bottom:.10f}")
    if abs(zfar_minus_znear) < 1e-6:
        import warnings
        warnings.warn(f"zfar - znear is too small ({zfar_minus_znear}), adjusting zfar")
        zfar_old = zfar
        zfar = znear + 1e-3  # 确保 zfar > znear
        zfar_minus_znear = zfar - znear
        # print(f"  WARNING: zfar adjusted from {zfar_old:.10f} to {zfar:.10f}")

    # 最终安全检查：确保分母不为 0
    if abs(right_minus_left) < 1e-10:
        # print(f"  CRITICAL: right_minus_left still too small ({right_minus_left}), forcing to 1e-3")
        right_minus_left = 1e-3
    if abs(top_minus_bottom) < 1e-10:
        # print(f"  CRITICAL: top_minus_bottom still too small ({top_minus_bottom}), forcing to 1e-3")
        top_minus_bottom = 1e-3
    if abs(zfar_minus_znear) < 1e-10:
        # print(f"  CRITICAL: zfar_minus_znear still too small ({zfar_minus_znear}), forcing to 1e-3")
        zfar_minus_znear = 1e-3
    
    # print(f"  Final values: right_minus_left={right_minus_left:.10f}, top_minus_bottom={top_minus_bottom:.10f}, zfar_minus_znear={zfar_minus_znear:.10f}")
    # print(f"  Computing P[0,0] = 2.0 * {znear} / {right_minus_left} = {2.0 * znear / right_minus_left:.10f}")

    P[0, 0] = 2.0 * znear / right_minus_left
    P[1, 1] = 2.0 * znear / top_minus_bottom
    P[0, 2] = (right + left) / right_minus_left
    P[1, 2] = (top + bottom) / top_minus_bottom
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / zfar_minus_znear
    P[2, 3] = -(zfar * znear) / zfar_minus_znear
    return P