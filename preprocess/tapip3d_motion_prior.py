import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.linalg import svd, norm
from sklearn.cluster import KMeans
import json
from utils.metrics import eval_axis_and_state_all, read_gt, read_joint_priori
from copy import deepcopy


def downsample_traj_spatial(trajectory, vis_mask, voxel_size=0.01, min_voxel_size=0.001):
    """
    Downsample trajectory by voxel grid sampling
    Args:
        trajectory: [T, 3] array
        vis_mask: [T] array
        voxel_size: Voxel size (relative to trajectory range)
    Returns:
        sampled_trajectory: [T', 3] array
    """
    min_bounds = np.min(trajectory, axis=0)
    max_bounds = np.max(trajectory, axis=0)
    
    ranges = max_bounds - min_bounds
    ranges = np.maximum(ranges, 1e-6)
    actual_voxel_size = max(ranges.max() * voxel_size, min_voxel_size)
    
    voxel_indices = np.floor((trajectory - min_bounds) / actual_voxel_size).astype(int) 
    voxel_dict = {}
    for i, idx in enumerate(voxel_indices):
        voxel_key = tuple(idx)
        if voxel_key not in voxel_dict and vis_mask[i]:
            voxel_dict[voxel_key] = i
    sampled_indices = list(voxel_dict.values())
    sampled_trajectory = trajectory[sampled_indices, :]
    return sampled_trajectory


def sample_valid_trajectory(trajectory, vis_mask, voxel_size=0.01, min_voxel_size=0.001):
    """
    Sample valid trajectory by
    1. Remove unvisible trajectory points
    2. Adaptive voxel grid sampling, suitable for trajectories with arbitrary ranges
    
    Args:
        trajectory: Point trajectory with shape [T, N, 3]
        vis_mask: Visibility mask with shape [T, N]
        voxel_size: Voxel size (relative to trajectory range)
    Returns:
        sampled_trajectory: List of N sampled trajectory points with shape [T', 3]
        downsample_ratios: List of N downsample ratios
    """
    T, N, _ = trajectory.shape
    sampled_trajectory = []
    downsample_ratios = []
    for n in range(N):
        traj = downsample_traj_spatial(trajectory[:, n, :], vis_mask[:, n], voxel_size, min_voxel_size)
        sampled_trajectory.append(traj)
        downsample_ratios.append(len(traj) / T)
    return sampled_trajectory, downsample_ratios


def fit_plane(points):
    centroid = np.mean(points, axis=0)
    _, _, vh = svd(points - centroid)
    normal = vh[-1]
    return centroid, normal


def project_onto_plane(points, centroid, normal):
    normal = normal / norm(normal)
    return points - ((points - centroid) @ normal)[:, None] * normal


def fit_circle_2d(points_2d):
    A = np.hstack((2 * points_2d, np.ones((points_2d.shape[0], 1))))
    b = np.sum(points_2d**2, axis=1)
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    center = x[:2]
    radius = np.sqrt(np.sum(center**2) + x[2])
    return center, radius


def cal_angular_diff(points, origin):
    o2t = points - origin[None]
    o2t_norm = np.linalg.norm(o2t, axis=-1)
    o2t_unit = o2t / (o2t_norm[..., None] + 1e-6)
    angular_diff = np.linalg.norm(np.diff(o2t_unit, axis=0), axis=-1)
    return angular_diff # [T-1]


def check_rigid_rotation(points, tol=0.05):
    centroid, normal = fit_plane(points)
    projected = project_onto_plane(points, centroid, normal)

    u = np.array([1.0, 0.0, 0.0])
    if np.abs(np.dot(u, normal)) > 0.9:
        u = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(normal, u)
    x_axis /= norm(x_axis)
    y_axis = np.cross(normal, x_axis)

    basis = np.stack([x_axis, y_axis], axis=1)
    points_2d = (projected - centroid) @ basis

    center_2d, radius = fit_circle_2d(points_2d)

    origin = centroid + basis @ center_2d
    errors = np.abs(norm(points - origin, axis=1) - radius)

    # calculate the max rotation angle
    threshold = np.cos(np.pi / 6)
    arrow = points - origin
    arrow = arrow / norm(arrow, axis=1)[:, None]
    cos_sim = np.dot(arrow, arrow.T) # [T, T]
    min_cos_sim = np.min(cos_sim)

    # is_valid = np.all(errors < tol)
    is_valid = np.all(errors < tol) and min_cos_sim < threshold
    return {
        "is_rigid": is_valid,
        "direction": normal / norm(normal),
        "origin": origin,
        "radius": radius,
        "normal": normal,
        "mean_error": np.mean(errors),
        "max_error": np.max(errors)
    }


def line_fit_ransac(trajectory, distance_threshold=0.05):
    T = trajectory.shape[0]
    best_inlier_count = 0
    best_direction = None
    best_inliers = None
    
    for _ in range(100):  # Number of RANSAC iterations
        # Randomly select 2 points
        idx = np.random.choice(T, 2, replace=False)
        p1, p2 = trajectory[idx]
        
        # Calculate line direction from these points
        direction = p2 - p1
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm > 1e-10:  # Avoid zero division
            direction = direction / direction_norm
            
            # Project all points onto this line
            centered_to_p1 = trajectory - p1
            projections = np.dot(centered_to_p1, direction)
            projected = p1 + np.outer(projections, direction)
            distances = np.linalg.norm(trajectory - projected, axis=1)
            
            # Count inliers
            curr_inliers = distances < distance_threshold
            inlier_count = np.sum(curr_inliers)
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_direction = direction
                best_inliers = curr_inliers
    return best_direction, best_inliers, best_inlier_count


def fit_line_to_trajectory(trajectory, distance_threshold=0.05, min_inlier_ratio=0.8, use_ransac=True):
    """
    Fit a line to trajectory points in 3D space, without assuming linear relationship with time
    
    Args:
        trajectory: Point trajectory with shape [T, 3]
        distance_threshold: Maximum distance from point to line to be considered an inlier
        min_inlier_ratio: Minimum required ratio of inliers
    
    Returns:
        success: Whether fitting succeeded
        line_params: Dictionary containing line parameters
        error: Average fitting error
    """
    T = trajectory.shape[0]
    
    try:
        # 1. PCA-based line fitting
        # Center the data
        mean_point = np.mean(trajectory, axis=0)
        centered = trajectory - mean_point
        
        # Perform SVD to find the principal direction (equivalent to PCA)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        
        # First singular vector corresponds to the direction of maximum variance
        line_direction = Vt[0]
        
        # 2. Calculate distances from each point to the fitted line
        # Project points onto the line direction
        projections = np.dot(centered, line_direction)
        
        # Reconstruct the projected points on the line
        projected_points = np.outer(projections, line_direction) + mean_point
        
        # Calculate perpendicular distances from original points to line
        distances = np.linalg.norm(trajectory - projected_points, axis=1)
        
        # 3. Determine inliers based on distance threshold
        inliers = distances < distance_threshold
        inlier_ratio = np.sum(inliers) / T
        
        # 4. Optional: Refine with RANSAC for more robustness
        if use_ransac and inlier_ratio < min_inlier_ratio and T >= 10:
            # Try RANSAC if simple PCA doesn't give enough inliers
            best_direction, best_inliers, best_inlier_count = line_fit_ransac(trajectory, distance_threshold)
            # Update with best RANSAC result if better than PCA
            if best_inlier_count / T >= inlier_ratio:
                line_direction = best_direction
                inliers = best_inliers
                inlier_ratio = best_inlier_count / T
                
                # Recalculate projections with best direction
                centered_to_mean = trajectory - mean_point
                projections = np.dot(centered_to_mean, line_direction)
                projected_points = mean_point + np.outer(projections, line_direction)
                distances = np.linalg.norm(trajectory - projected_points, axis=1)
        
        # 5. Check if fitting was successful
        if inlier_ratio >= min_inlier_ratio:
            # Calculate error on inliers only
            if np.sum(inliers) > 0:
                mean_error = np.mean(distances[inliers])
            else:
                mean_error = float('inf')
            
            # Calculate line endpoints for visualization
            # Find the range of projections
            min_proj = np.min(projections)
            max_proj = np.max(projections)
            
            # Calculate endpoints
            start_point = mean_point + min_proj * line_direction
            end_point = mean_point + max_proj * line_direction
            line_length = np.linalg.norm(end_point - start_point)
            # Return success with line parameters
            line_params = {
                'origin': np.zeros(3),            
                'direction': line_direction,      # Direction vector
                'start_point': start_point,       # Start endpoint of the line segment
                'end_point': end_point,           # End endpoint of the line segment
                'length': line_length,            # Length of the line segment
                'inlier_ratio': inlier_ratio,     # Ratio of points that fit the line
                'inlier_mask': inliers,
                'mean_error': mean_error            # Boolean mask of inliers
            }
            
            return True, line_params, mean_error
        else:
            return False, None, float('inf')
        
    except Exception as e:
        # print(f"Spatial line fitting failed: {e}")
        return False, None, float('inf')
    

def fit_circle_to_trajectory(trajectory):
    """
    Fit a circle to a trajectory to determine if it represents rotational motion
    
    Args:
        trajectory: Point trajectory with shape [T, 3]
    Returns:
        success: Whether fitting succeeded
        circle_params: Dictionary containing circle parameters (center, radius, normal)
        error: Average fitting error
    """
    results = check_rigid_rotation(trajectory)
    return results["is_rigid"], results, results["mean_error"]


def vis_trajectory(trajectory, save_path=None):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color='black', alpha=0.8, linewidth=1)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def classify_trajectory(trajectory, line_threshold=0.05, circle_threshold=0.05, line_first=False):
    """
    Classify a single point trajectory into one of three motion types:
    static, translation (line), or rotation (circle)
    
    Args:
        trajectory: Point trajectory with shape [T, 3]
        line_threshold: Maximum error to consider as translational motion
        circle_threshold: Maximum error to consider as rotational motion
    
    Returns:
        motion_type: Integer (-1=unknown, 0=static, 1=translation, 2=rotation)
        motion_params: Parameters of the detected motion
        motion_error: Error of the detected motion
    """
    # Fit line to trajectory
    line_success, line_params, line_error = fit_line_to_trajectory(trajectory)
    if line_first and line_success and line_error < line_threshold:
        return 1, line_params, line_error
    
    # Fit circle to trajectory
    circle_success, circle_params, circle_error = fit_circle_to_trajectory(trajectory)
    
    # Determine motion type based on fitting errors
    ratio = 2 * line_threshold / circle_threshold
    if line_success and line_error < line_threshold and (not circle_success or line_error < ratio * circle_error):
        return 1, line_params, line_error
    elif circle_success and circle_error < circle_threshold:
        return 2, circle_params, circle_error
    else:
        return -1, None, min(float('inf'), line_error, circle_error)


def weighted_mean(track, vis_mask):
    # track: [T, N, 3], vis_mask: [T, N]
    mu = np.sum(track * vis_mask[:, :, None], axis=0) / (np.sum(vis_mask, axis=0)[:, None] + 1e-6) # [N, 3]
    return mu


def identify_static_points(track, vis_mask, static_threshold=0.01):
    """
    Identify static points in the trajectory
    track: [T, N, 3]
    vis_mask: [T, N]
    """
    mu = weighted_mean(track, vis_mask) # [N, 3]
    dist = np.linalg.norm(track - mu[None], axis=-1) # [T, N]
    dist[~vis_mask] = 0
    max_dist = np.max(dist, axis=0)
    static_mask = max_dist < static_threshold * max_dist.max() # [N]
    static_idx = np.nonzero(static_mask)[0]
    track[:, static_idx] = mu[static_idx]
    return track, static_mask


def select_plausible_motion_tracks(tapip3d_tracks, visibility_mask, static_threshold=0.01, line_threshold=0.05, circle_threshold=0.05, line_first=False):
    """
    Classify trajectories and filter out points with unreasonable motion
    
    Args:
        tapip3d_tracks: Point trajectories with shape [T, N, 3]
        static_threshold: Maximum displacement to consider as static
        line_threshold: Maximum error to consider as translational motion
        circle_threshold: Maximum error to consider as rotational motion
    
    Returns:
        valid_motion: Boolean mask of shape [N] indicating points with valid motion
        motion_types: Integer array of shape [N] with motion classifications
        motion_params: List of motion parameters for each point
    """
    T, N, _ = tapip3d_tracks.shape
    
    track_motion_labels = np.zeros(N, dtype=int)
    track_motion_fits = [None] * N
    motion_errors = np.zeros(N)
    
    tapip3d_tracks, static_mask = identify_static_points(tapip3d_tracks, visibility_mask, static_threshold) # [N]

    sampled_tracks, downsample_ratios = sample_valid_trajectory(tapip3d_tracks, visibility_mask, 0.02)
    downsample_ratios = np.array(downsample_ratios)
    # Analyze each point's trajectory individually
    for n in range(N):
        if static_mask[n]:
            track_motion_labels[n] = 0
            track_motion_fits[n] = {}
            motion_errors[n] = 0
        elif downsample_ratios[n] < 0.1 and visibility_mask[:, n].sum() < 0.2 * T: # too few valid points
            track_motion_labels[n] = -1
            track_motion_fits[n] = None
            motion_errors[n] = None
        else:
            track_motion_labels[n], track_motion_fits[n], motion_errors[n] = classify_trajectory(
                sampled_tracks[n],
                line_threshold=line_threshold,
                circle_threshold=circle_threshold,
                line_first=line_first
            )
    
    # Points with valid motion are those classified as one of the known types
    valid_motion = track_motion_labels > -1
    
    return tapip3d_tracks, valid_motion, track_motion_labels, track_motion_fits


def visualize_motion_types(trajectories, motion_types, motion_params, save_path=None):
    """
    Visualize trajectories colored by motion type
    
    Args:
        trajectories: Point trajectories with shape [T, N, 3]
        motion_types: Integer array of shape [N] with motion classifications
        save_path: Optional path to save the visualization
    """
    T, N, _ = trajectories.shape
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=(12, 9), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_box_aspect([1, 1, 1])
    
    colors = {
        -1: '#8B5A96',
        0: '#2C3E50',
        1: '#27AE60',
        2: '#E74C3C'
    }
    
    # ax.set_facecolor('#F8F9FA')
    ax.set_facecolor('#000000')
    fig.patch.set_facecolor('#000000')
    
    for n in range(N):
        if motion_types[n] == -1:
            continue
        color = colors[motion_types[n]]
        if motion_types[n] == 2 and (np.abs(motion_params[n]["origin"]) > 2).any():
            color = '#F39C12'  
        ax.plot(trajectories[:, n, 0], trajectories[:, n, 1], trajectories[:, n, 2], 
                color=color, alpha=0.7, linewidth=0.8)

    static_points = trajectories[:, motion_types==0]
    if static_points.size > 0:
        ax.scatter(static_points[0, :, 0], static_points[0, :, 1], static_points[0, :, 2], 
                   c='#2C3E50', alpha=0.3, s=2, edgecolors='none')
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=colors[-1], lw=2.5, label='Noise'),
        Line2D([0], [0], color=colors[0], lw=2.5, label='Static'),
        Line2D([0], [0], color=colors[1], lw=2.5, label='Translation'),
        Line2D([0], [0], color=colors[2], lw=2.5, label='Rotation')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
              fancybox=True, shadow=True, fontsize=10)
    
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z', fontsize=12, fontweight='bold')
    ax.set_title('Trajectory Motion Type Classification', fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    ax.xaxis.line.set_color('#BDC3C7')
    ax.yaxis.line.set_color('#BDC3C7')
    ax.zaxis.line.set_color('#BDC3C7')
    ax.xaxis.line.set_linewidth(1.5)
    ax.yaxis.line.set_linewidth(1.5)
    ax.zaxis.line.set_linewidth(1.5)
    
    ax.grid(True, alpha=1, color='#000000', linewidth=0.5)
    
    ax.view_init(elev=20, azim=160)
    
    x_min, x_max = trajectories[:, :, 0].min(), trajectories[:, :, 0].max()
    y_min, y_max = trajectories[:, :, 1].min(), trajectories[:, :, 1].max()
    z_min, z_max = trajectories[:, :, 2].min(), trajectories[:, :, 2].max()
    
    margin = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
    ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
    ax.set_zlim(z_min - margin * z_range, z_max + margin * z_range)
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, facecolor='white', pad_inches=0.1)
    
    for n in range(N):
        if motion_types[n] == -1:
            color = colors[motion_types[n]]
            ax.plot(trajectories[:, n, 0], trajectories[:, n, 1], trajectories[:, n, 2], 
                    color=color, alpha=0.7, linewidth=0.8)
    
    if save_path:
        plt.savefig(save_path.replace('.png', '_all.png'), dpi=300, facecolor='white', pad_inches=0.1)
    
    plt.close()


def visualize_fitted_models(trajectories, motion_types, motion_params, indices=None, save_path=None):
    """
    Visualize fitted models (lines and circles) for selected trajectories
    
    Args:
        trajectories: Point trajectories with shape [T, N, 3]
        motion_types: Integer array of shape [N] with motion classifications
        motion_params: List of motion parameters for each point
        indices: Optional list of point indices to visualize
        save_path: Optional path to save the visualization
    """
    T, N, _ = trajectories.shape
    
    if indices is None:
        static_indices = np.random.choice(np.where(motion_types == 1)[0], size=min(100, np.sum(motion_types == 1)))
        transl_indices = np.random.choice(np.where(motion_types == 2)[0], size=min(100, np.sum(motion_types == 2)))
        rot_indices = np.random.choice(np.where(motion_types == 3)[0], size=min(100, np.sum(motion_types == 3)))
        indices = np.concatenate([static_indices, transl_indices, rot_indices])
    
    fig = plt.figure(figsize=(12, 9), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#000000')
    fig.patch.set_facecolor('#000000')
    
    # Color map for motion types
    colors = {
        -1: '#8B5A96',
        0: '#2C3E50',
        1: '#27AE60',
        2: '#E74C3C'
    }
    # ax.set_box_aspect([1, 1, 1])
    
    ax.view_init(elev=20, azim=160)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    ax.xaxis.line.set_color('#BDC3C7')
    ax.yaxis.line.set_color('#BDC3C7')
    ax.zaxis.line.set_color('#BDC3C7')
    ax.xaxis.line.set_linewidth(1.5)
    ax.yaxis.line.set_linewidth(1.5)
    ax.zaxis.line.set_linewidth(1.5)
    for idx in indices:
        motion_type = motion_types[idx]
        traj = trajectories[:, idx, :]
        color = colors[motion_type]
        
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                color=color, alpha=0.8, linewidth=2, label=f'Point {idx}')
        
        # Plot fitted model
        if motion_type == 0:  # Static
            mean_pos = traj.mean(axis=0)
            ax.scatter([mean_pos[0]], [mean_pos[1]], [mean_pos[2]], 
                       color=color, s=100, marker='o', edgecolors='black')
            
        elif motion_type == 1:  # Translation (Line)
            # Plot fitted line
            start_point = motion_params[idx]['start_point']
            end_point = motion_params[idx]['end_point']
            line_points = np.stack([start_point, end_point], axis=0)
            ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 
                   color=color, linestyle='--', linewidth=1)
            
        elif motion_type == 2:  # Rotation (Circle)
            # Plot fitted circle
            center = motion_params[idx]['origin']
            radius = motion_params[idx]['radius']
            normal = motion_params[idx]['normal']
            
            # Plot center and normal vector
            ax.scatter([center[0]], [center[1]], [center[2]], 
                      color=color, s=100, marker='x')
            # Plot normal vector
            arrow_length = radius * 0.5
            ax.quiver(center[0], center[1], center[2], 
                     normal[0], normal[1], normal[2], 
                     length=arrow_length, color=color, arrow_length_ratio=0.1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Fitted Models for Selected Trajectories')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.savefig('fitted_models_visualization.png')
    
    plt.close()


def render_joint_prior_debug(joint_details, save_path=None):
    """
    Visualize generated joint details
    """
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    for joint_detail in joint_details:
        # random color
        color = np.random.rand(3)
        if joint_detail["joint_type"] == 's':
            center = np.array(joint_detail['center'])
            dist_max = joint_detail['dist_max']
            ax.scatter([center[0]], [center[1]], [center[2]], 
                       color=color, s=100, marker='o', edgecolors='black')
            # rn = np.random.randn(100, 3)
            # rn = rn / np.linalg.norm(rn, axis=-1, keepdims=True)
            # circle_points = center + rn * dist_max
            # ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], 
            #         color=color, linewidth=1)
        elif joint_detail['joint_type'] == 'p':
            center = np.array(joint_detail['center'])
            dist_max = joint_detail['dist_max']
            direction = np.array(joint_detail['direction'])
            ax.scatter([center[0]], [center[1]], [center[2]], 
                       color=color, s=100, marker='o', edgecolors='black')
            # rn = np.random.randn(100, 3)
            # rn = rn / np.linalg.norm(rn, axis=-1, keepdims=True)
            # circle_points = center + rn * dist_max
            # ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], 
            #         color=color, linewidth=1)
            start_point = center
            end_point = center + direction * dist_max
            line_points = np.stack([start_point, end_point], axis=0)
            ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 
                   color=color, linestyle='--', linewidth=1)
        elif joint_detail['joint_type'] == 'r':
            center = np.array(joint_detail['center'])
            dist_max = joint_detail['dist_max']
            direction = np.array(joint_detail['direction'])
            origin = np.array(joint_detail['origin'])
            ax.scatter([center[0]], [center[1]], [center[2]], 
                       color=color, s=100, marker='o', edgecolors='black')
            # rn = np.random.randn(100, 3)
            # rn = rn / np.linalg.norm(rn, axis=-1, keepdims=True) * 0.01
            # circle_points = center + rn * dist_max
            # ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], 
            #         color=color, linewidth=1)
            # plot origin and direction
            ax.scatter([origin[0]], [origin[1]], [origin[2]], 
                       color=color, s=100, marker='x')
            ax.quiver(origin[0], origin[1], origin[2], 
                     direction[0], direction[1], direction[2], 
                     length=dist_max, color=color, arrow_length_ratio=0.1)
    plt.savefig(save_path)
    plt.close()


def print_motion_prior_summary(motion_types):
    """
    Print statistics about detected motion types
    
    Args:
        motion_types: Integer array of shape [N] with motion classifications
    """
    N = len(motion_types)
    
    n_unknown = np.sum(motion_types == -1)
    n_static = np.sum(motion_types == 0)
    n_translation = np.sum(motion_types == 1)
    n_rotation = np.sum(motion_types == 2)
    
    print(f"Motion Type Statistics:")
    print(f"  Static points: {n_static} ({n_static/N*100:.1f}%)")
    print(f"  Translational points: {n_translation} ({n_translation/N*100:.1f}%)")
    print(f"  Rotational points: {n_rotation} ({n_rotation/N*100:.1f}%)")
    print(f"  Noise motion: {n_unknown} ({n_unknown/N*100:.1f}%)")
    print(f"  Total valid motion: {N-n_unknown} ({(N-n_unknown)/N*100:.1f}%)") 


def filter_cluster_outliers(features, labels, centers, direction_slice=(6,9), angle_threshold_deg=30, dist_std_ratio=3):
    """
    For each cluster, first do direction angle filtering, 
    then do Euclidean distance filtering, and return the filtered features and labels.
    Args:
        features: [N, D]
        labels: [N]
        centers: [K, D]
        direction_slice: tuple, direction feature index
        angle_threshold_deg: angle threshold (degree)
        dist_std_ratio: standard deviation ratio for distance filtering
    """
    angle_threshold_cos = np.cos(np.deg2rad(angle_threshold_deg))
    n_clusters = centers.shape[0]
    valid_mask = np.zeros(features.shape[0], dtype=bool)
    for i in range(n_clusters):
        idxs = np.where(labels == i)[0]
        if len(idxs) == 0:
            continue
        feats = features[idxs]
        center = centers[i]
        # direction
        directions = feats[:, direction_slice[0]:direction_slice[1]]
        center_dir = center[direction_slice[0]:direction_slice[1]]
        directions = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8)
        center_dir = center_dir / (np.linalg.norm(center_dir) + 1e-8)
        cos_angles = np.dot(directions, center_dir)
        angle_mask = cos_angles > angle_threshold_cos

        # Euclidean distance filtering
        dists = np.linalg.norm(feats[:, :3] - center[:3], axis=1)
        mean_dist = dists.mean()
        std_dist = dists.std()
        dist_threshold = mean_dist + dist_std_ratio * std_dist
        dist_mask = dists < dist_threshold
        valid = angle_mask & dist_mask
        valid_mask[idxs] = valid
    return valid_mask


def cluster_motion_priors(features, n_clusters, n_iter=3, angle_threshold_deg=50, dist_std_ratio=3, direction_slice=(6, 9)):
    if features.shape[0] == 0:
        return np.zeros(0, dtype=bool), np.zeros((n_clusters, 0)), np.zeros(0, dtype=int) - 1

    valid_features = features
    valid_mask = np.ones(features.shape[0], dtype=bool)
    valid_labels = np.ones(features.shape[0], dtype=int) * -1
    centers = None
    for i in range(n_iter):
        current_idxs = np.where(valid_mask)[0]
        if len(current_idxs) == 0:
            print("Warning: cluster filtering removed all samples; keeping previous valid clusters.")
            valid_mask[:] = valid_labels >= 0
            break
        valid_features = features[valid_mask]
        n_eff_clusters = min(n_clusters, valid_features.shape[0])
        if n_eff_clusters == 1:
            labels = np.zeros(valid_features.shape[0], dtype=int)
            centers = valid_features.mean(axis=0, keepdims=True)
        else:
            kmeans = KMeans(n_clusters=n_eff_clusters, random_state=0)
            kmeans.fit(valid_features)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_
        if centers.shape[0] < n_clusters:
            pad = np.repeat(centers[-1:], n_clusters - centers.shape[0], axis=0)
            centers = np.concatenate([centers, pad], axis=0)
        valid_labels[current_idxs] = labels
        mask = filter_cluster_outliers(valid_features, labels, centers, direction_slice, angle_threshold_deg, dist_std_ratio)
        if not mask.any():
            print("Warning: cluster outlier filtering would remove all samples; stop filtering.")
            break
        angle_threshold_deg = angle_threshold_deg - 10
        valid_mask[valid_mask] = mask
        valid_labels[valid_mask] = labels[mask]
    return valid_mask, centers, valid_labels

def build_tapip3d_motion_prior(scene_name, data_path, n_query_frames=4, use_vis_mask=False, visualize=True, print_info=True, realscan=False):
    tapip3d_result = np.load(f"{data_path}/{scene_name}/{scene_name}.n{n_query_frames}.npz")
    tapip3d_tracks = tapip3d_result["coords"]
    visibility_mask = tapip3d_result["visibs"]
    T, N, _ = tapip3d_tracks.shape
    if not use_vis_mask:
        visibility_mask = np.ones_like(visibility_mask)

    joint_priori = read_joint_priori(f'{data_path}/{scene_name}/joint_priori.json')
    joint_types = [joint_prior["joint_type"] for joint_prior in joint_priori]
    # joint_types = ['r', 'p', 'p', 'p']
    # joint_types = ['p', 'p', 'p']
    # joint_types = ['r', 'r', 'r']
    n_prismatic_joints = len([t for t in joint_types if t == "p"])
    n_revolute_joints = len([t for t in joint_types if t == "r"])

    eps_s, eps_l, eps_r = 0.1, 0.01, 0.01
    if realscan:
        eps_s, eps_l, eps_r = 0.25, 0.01, 0.01
        if n_prismatic_joints == 0:
            eps_r = 0.02
    tapip3d_tracks, plausible_motion_mask, track_motion_labels, track_motion_fits = select_plausible_motion_tracks(
        tapip3d_tracks, 
        visibility_mask,
        static_threshold=eps_s, 
        line_threshold=eps_l, 
        circle_threshold=eps_r,
    )
    if n_prismatic_joints == 0:
        plausible_motion_mask[track_motion_labels == 1] = False
        track_motion_labels[track_motion_labels == 1] = -1
    if n_revolute_joints == 0:
        plausible_motion_mask[track_motion_labels == 2] = False
        track_motion_labels[track_motion_labels == 2] = -1
    

    if print_info:
        print_motion_prior_summary(track_motion_labels)

    revolute_reference_direction = None
    prismatic_reference_direction = None
    static_features = []
    revolute_features = []
    prismatic_features = []
    line_lengths = []
    raw_part_ids = []
    for n in range(tapip3d_tracks.shape[1]):
        track = tapip3d_tracks[:, n, :]
        start_pos = track[0]
        mean_pos = track.mean(axis=0)
        if track_motion_labels[n] == 0: # static
            static_features.append(start_pos)
            raw_part_ids.append(0)
        elif track_motion_labels[n] == 1: # prismatic
            direction = track_motion_fits[n]["direction"]
            line_length = track_motion_fits[n]["length"]
            line_lengths.append(line_length)
            delta_traj = np.diff(track, axis=0)
            delta_traj = np.linalg.norm(delta_traj, axis=-1) / line_length * 5
            if prismatic_reference_direction is None:
                prismatic_reference_direction = direction
            direction = np.where(direction.dot(prismatic_reference_direction[None].T) < 0, -direction, direction)
            feature = np.concatenate([start_pos, mean_pos, direction, delta_traj.reshape(-1)])
            prismatic_features.append(feature)
            raw_part_ids.append(1)
        elif track_motion_labels[n] == 2: # revolute
            origin = track_motion_fits[n]["origin"]
            direction = track_motion_fits[n]["direction"]
            delta_traj = cal_angular_diff(track, origin)
            if revolute_reference_direction is None:
                revolute_reference_direction = direction
            direction = np.where(direction.dot(revolute_reference_direction[None].T) < 0, -direction, direction)
            feature = np.concatenate([start_pos, mean_pos, direction, origin, delta_traj.reshape(-1)])
            revolute_features.append(feature)
            raw_part_ids.append(n_prismatic_joints + 1)
        else:
            raw_part_ids.append(-1)
    raw_part_ids = np.array(raw_part_ids)
    clustered_part_ids = deepcopy(raw_part_ids)
    static_features = np.array(static_features)
    prismatic_features = np.array(prismatic_features)
    revolute_features = np.array(revolute_features)
    scene_extent = np.linalg.norm(np.nanmax(tapip3d_tracks[0], axis=0) - np.nanmin(tapip3d_tracks[0], axis=0))
    min_dist_max = max(scene_extent * 0.01, 1e-3)

    if n_prismatic_joints > 0:
        mean_line_length = np.mean(line_lengths)
        prismatic_features[:, :6] = prismatic_features[:, :6] / mean_line_length
        p_mask = raw_part_ids == 1
        p_valid_mask, p_centers, p_labels = cluster_motion_priors(prismatic_features, n_prismatic_joints)
        clustered_part_ids[p_mask] += p_labels
        # plausible_motion_mask[p_mask] &= p_valid_mask
        prismatic_features[:, :6] = prismatic_features[:, :6] * mean_line_length
        p_centers[:, :6] = p_centers[:, :6] * mean_line_length
    if n_revolute_joints > 0:
        r_mask = raw_part_ids == n_prismatic_joints + 1
        r_valid_mask, r_centers, r_labels = cluster_motion_priors(revolute_features, n_revolute_joints)
        clustered_part_ids[r_mask] += r_labels
        # plausible_motion_mask[r_mask] &= r_valid_mask

    static_features = np.nan_to_num(static_features)
    static_center = np.mean(static_features, axis=0)
    dist = np.linalg.norm(static_features - static_center[None], axis=-1)
    dist_max = max(dist.max() * 1.0, min_dist_max)
    joint_details = [
        {
            "joint_type": "s",
            "center": static_center.tolist(),
            "dist_max": float(dist_max),
            "direction": [0, 0, 0],
            "origin": [0, 0, 0],
        }
    ]
    if n_prismatic_joints > 0:
        for n_p in range(n_prismatic_joints):
            start_pos = prismatic_features[p_labels == n_p][:, :3]
            center = p_centers[n_p]
            dist = np.linalg.norm(start_pos - center[:3][None], axis=-1)
            dist_max = max((dist.max() * 0.2) if dist.size > 0 else 0.0, min_dist_max)
            joint_details.append({
                "joint_type": "p",
                "center": center[:3].tolist(),
                "dist_max": float(dist_max),
                "direction": center[6:9].tolist(),
                "origin": [0, 0, 0]
            })
    if n_revolute_joints > 0:
        for n_r in range(n_revolute_joints):
            start_pos = revolute_features[r_labels == n_r][:, :3]
            center = r_centers[n_r]
            dist = np.linalg.norm(start_pos - center[:3][None], axis=-1)
            dist_max = max((dist.max() * 0.2) if dist.size > 0 else 0.0, min_dist_max)
            joint_details.append({
                "joint_type": "r",
                "center": center[:3].tolist(),
                "dist_max": float(dist_max),
                "direction": center[6:9].tolist(),
                "origin": center[9:12].tolist()
            })
    with open(f"{data_path}/{scene_name}/joint_details.json", "w") as f:
        json.dump(joint_details, f, indent=4)
    
    selected_tracks = tapip3d_tracks[:, plausible_motion_mask]
    selected_visibility = visibility_mask[:, plausible_motion_mask]
    selected_part_ids = clustered_part_ids[plausible_motion_mask]
    np.savez(f"{data_path}/{scene_name}/trajectory_tapip3d.npz", 
             coords=selected_tracks, 
             visibs=selected_visibility,
            #  mask_ids=selected_part_ids,
            #  video=tapip3d_result["video"],
            #  depths=tapip3d_result["depths"],
            #  intrinsics=tapip3d_result["intrinsics"],
            #  extrinsics=tapip3d_result["extrinsics"],
             )
    np.savez(f"{data_path}/{scene_name}/trajectory_tapip3d_visualization.npz", 
             coords=selected_tracks, 
             visibs=selected_visibility,
             mask_ids=selected_part_ids,
             video=tapip3d_result["video"],
             depths=tapip3d_result["depths"],
             intrinsics=tapip3d_result["intrinsics"],
             extrinsics=tapip3d_result["extrinsics"],
             )
    if visualize:
        visualize_motion_types(tapip3d_tracks, track_motion_labels, track_motion_fits, save_path=f"{data_path}/{scene_name}/motion.png")
        visualize_fitted_models(tapip3d_tracks, track_motion_labels, track_motion_fits, save_path=f"{data_path}/{scene_name}/fit_model.png")
        render_joint_prior_debug(joint_details, save_path=f"{data_path}/{scene_name}/joints.png")
        # visualize_trajectory_examples(tapip3d_tracks, track_motion_labels, track_motion_fits, save_dir=f"{data_path}/{scene_name}/trajectory_examples")

    # evaluate
    try:
        pred_joint_list = joint_details[1:]
        gt_joint_details = read_gt(f'{data_path}/{scene_name}/gt/mobility_v2.json')
        results, perm = eval_axis_and_state_all(pred_joint_list, gt_joint_details)
        results = np.round(np.array(results), 2)
        return results
    except:
        return np.zeros((4, 2))


def visualize_trajectory_examples(trajectories, motion_types, motion_params, save_dir=None, n_examples=10):
    """
    visualize trajectory examples
    Args:
        trajectories: trajectory data [T, N, 3]
        motion_types: motion type array [N]
        motion_params: motion parameter list [N]
        save_dir: save directory
        n_examples: number of examples for each motion type
    """
    T, N, _ = trajectories.shape
    
    motion_categories = {0: 'static', 1: 'prismatic', 2: 'revolute', -1: 'noise'}

    x_min, x_max = trajectories[:, 0].min(), trajectories[:, 0].max()
    y_min, y_max = trajectories[:, 1].min(), trajectories[:, 1].max()
    z_min, z_max = trajectories[:, 2].min(), trajectories[:, 2].max()
    ranges = [x_min, x_max, y_min, y_max, z_min, z_max]
    
    for motion_type, category_name in motion_categories.items():
        type_indices = np.where(motion_types == motion_type)[0]
        
        if len(type_indices) == 0:
            print(f"No {category_name} trajectories found")
            continue
        
        if motion_type == 1 or motion_type == 2:
            mean_error = np.array([motion_params[id]['mean_error'] for id in type_indices])
            selected_indices = type_indices[np.argsort(mean_error)[:n_examples]]
        else:
            selected_indices = np.random.choice(type_indices, size=n_examples, replace=False)
        
        for i, idx in enumerate(selected_indices):
            trajectory = trajectories[:, idx, :]
            params = motion_params[idx] if motion_params[idx] is not None else None
            
            visualize_single_trajectory_fitting(
                trajectory, motion_type, params, 
                save_dir=save_dir, 
                trajectory_id=f"{category_name}_{i+1}",
                ranges=ranges
            )
    
    print(f"Generated {n_examples * len(motion_categories)} trajectory visualizations in {save_dir}")

def visualize_single_trajectory_fitting(trajectory, motion_type, motion_params, save_dir=None, trajectory_id=0, ranges=None):
    """
    visualize single trajectory and its fitting result
    
    Args:
        trajectory: single trajectory [T, 3]
        motion_type: motion type (0=static, 1=prismatic, 2=revolute, -1=noise)
        motion_params: motion parameter
        save_dir: save directory
        trajectory_id: trajectory ID, for file name
    """
    motion_names = {0: 'Static', 1: 'Prismatic', 2: 'Revolute', -1: 'Noise'}
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')

    # ax.set_xlabel('X', fontsize=12, fontweight='bold')
    # ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    # ax.set_zlabel('Z', fontsize=12, fontweight='bold')
    
    # ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.xaxis.line.set_color('#BDC3C7')
    ax.yaxis.line.set_color('#BDC3C7')
    ax.zaxis.line.set_color('#BDC3C7')
    ax.xaxis.line.set_linewidth(0.1)
    ax.yaxis.line.set_linewidth(0.1)
    ax.zaxis.line.set_linewidth(0.1)


    ax.grid(True, alpha=0.5, color='#BDC3C7', linewidth=0.8)
    
    ax.view_init(elev=20, azim=135)  
    x_min, x_max = trajectory[:, 0].min(), trajectory[:, 0].max()
    y_min, y_max = trajectory[:, 1].min(), trajectory[:, 1].max()
    z_min, z_max = trajectory[:, 2].min(), trajectory[:, 2].max()
    w = 0.05
    if ranges is not None:
        x_min, x_max = x_min - (ranges[1] - ranges[0]) * w, x_max + (ranges[1] - ranges[0]) * w
        y_min, y_max = y_min - (ranges[3] - ranges[2]) * w, y_max + (ranges[3] - ranges[2]) * w
        z_min, z_max = z_min - (ranges[5] - ranges[4]) * w, z_max + (ranges[5] - ranges[4]) * w
    
    z_min, z_max = z_min + 0.05, z_max + 0.05
    margin = 0.05
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
    ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
    ax.set_zlim(z_min - margin * z_range, z_max + margin * z_range)
    
    colors = {
        -1: '#8B5A96',
        0: '#2C3E50',
        1: '#27AE60',
        2: '#E74C3C'
    }

    if motion_type != 0:
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                color='#2C3E50', alpha=0.8, linewidth=2, label='Original Trajectory')
    elif motion_type == -1:
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                color='#8B5A96', alpha=0.8, linewidth=2, label='Original Trajectory')
    else:
        mean_pos = np.mean(trajectory, axis=0)
        ax.scatter([mean_pos[0]], [mean_pos[1]], [mean_pos[2]], 
                   color=colors[motion_type], s=200, marker='o', edgecolors='black', linewidth=2,
                   label='Static Point')
    
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        filename = f'origin_trajectory_{trajectory_id}_{motion_names[motion_type].lower()}.png'
        save_path = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, facecolor='white', pad_inches=0)
    
    
    if motion_type == 0:  # Static
        mean_pos = np.mean(trajectory, axis=0)
        ax.scatter([mean_pos[0]], [mean_pos[1]], [mean_pos[2]], 
                   color=colors[motion_type], s=200, marker='o', edgecolors='black', linewidth=2,
                   label='Static Point')
        
    elif motion_type == 1:  # Prismatic
        if motion_params is not None:
            start_point = motion_params['start_point']
            end_point = motion_params['end_point']
            line_points = np.stack([start_point, end_point], axis=0)
            ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 
                   color=colors[motion_type], linestyle='-', linewidth=3, alpha=0.8,
                   label='Fitted Line')
            
            direction = motion_params['direction']
            center = (start_point + end_point) / 2
            horizontal_direction = np.random.randn(3)
            horizontal_direction = np.cross(direction, horizontal_direction)
            horizontal_direction = horizontal_direction / np.linalg.norm(horizontal_direction)
            arrow_start = start_point + horizontal_direction * 0.03
            arrow_length = motion_params['length'] * 0.6
            arrow_end = center + direction * arrow_length
            
            ax.quiver(arrow_start[0], arrow_start[1], arrow_start[2], 
                     direction[0], direction[1], direction[2], 
                     length=arrow_length, color=colors[motion_type], 
                     arrow_length_ratio=0.1, linewidth=3, alpha=0.8,
                     label='Direction')
            
    elif motion_type == 2:  # Revolute
        if motion_params is not None:
            origin = motion_params['origin']
            direction = motion_params['direction']
            radius = motion_params['radius']
            
            ax.scatter([origin[0]], [origin[1]], [origin[2]], 
                       color='#E74C3C', s=200, marker='x', linewidth=3,
                       label='Rotation Center')
            
            arrow_length = radius * 0.3
            arrow_end = origin + direction * arrow_length
            
            ax.quiver(origin[0], origin[1], origin[2], 
                     direction[0], direction[1], direction[2], 
                     length=arrow_length, color=colors[motion_type], 
                     arrow_length_ratio=0.3, linewidth=3, alpha=0.8,
                     label='Rotation Axis')
            
            normal = direction / np.linalg.norm(direction)
            u = np.array([1.0, 0.0, 0.0])
            if np.abs(np.dot(u, normal)) > 0.9:
                u = np.array([0.0, 1.0, 0.0])
            x_axis = np.cross(normal, u)
            x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(normal, x_axis)
            
            theta = np.linspace(0, 2*np.pi, 50)
            circle_points = origin + radius * (np.outer(np.cos(theta), x_axis) + 
                                             np.outer(np.sin(theta), y_axis))
            
            ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], 
                   color=colors[motion_type], linestyle='--', linewidth=2, alpha=0.8,
                   label='Fitted Circle')
            
    elif motion_type == -1:  # Noise
        pass
    
    if save_dir:
        filename = f'trajectory_{trajectory_id}_{motion_names[motion_type].lower()}.png'
        save_path = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, facecolor='white', pad_inches=0)
    
    plt.close()
