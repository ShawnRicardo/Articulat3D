import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd, norm
from tqdm import tqdm


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

    is_valid = np.all(errors < tol)
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
            
            # Return success with line parameters
            line_params = {
                'origin': np.zeros(3),            
                'direction': line_direction,      # Direction vector
                'start_point': start_point,       # Start endpoint of the line segment
                'end_point': end_point,           # End endpoint of the line segment
                'inlier_ratio': inlier_ratio,     # Ratio of points that fit the line
                'inlier_mask': inliers            # Boolean mask of inliers
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

def classify_trajectory(trajectory, static_threshold=0.01, line_threshold=0.05, circle_threshold=0.05):
    """
    Classify a single point trajectory into one of three motion types:
    static, translation (line), or rotation (circle)
    
    Args:
        trajectory: Point trajectory with shape [T, 3]
        static_threshold: Maximum displacement to consider as static
        line_threshold: Maximum error to consider as translational motion
        circle_threshold: Maximum error to consider as rotational motion
    
    Returns:
        motion_type: Integer (0=unknown, 1=static, 2=translation, 3=rotation)
        motion_params: Parameters of the detected motion
        motion_error: Error of the detected motion
    """
    # Fit line to trajectory
    line_success, line_params, line_error = fit_line_to_trajectory(trajectory)
    if line_success and line_error < line_threshold:
        return 2, line_params, line_error
    
    # Fit circle to trajectory
    circle_success, circle_params, circle_error = fit_circle_to_trajectory(trajectory)
    
    # Determine motion type based on fitting errors
    if line_success and line_error < line_threshold and (not circle_success or line_error < circle_error):
        return 2, line_params, line_error
    elif circle_success and circle_error < circle_threshold and (not line_success or circle_error < line_error):
        return 3, circle_params, circle_error
    else:
        return 0, None, min(float('inf'), line_error, circle_error)

def weighted_mean(track, vis_mask):
    # track: [T, N, 3], vis_mask: [T, N]
    mu = np.sum(track * vis_mask[:, :, None], axis=0) / np.sum(vis_mask, axis=0)[:, None] # [N, 3]
    return mu

def identify_static_points(track, vis_mask):
    """
    Identify static points in the trajectory
    track: [T, N, 3]
    vis_mask: [T, N]
    """
    mu = weighted_mean(track, vis_mask) # [N, 3]
    dist = np.linalg.norm(track - mu[None], axis=-1) # [T, N]
    dist[~vis_mask] = 0
    max_dist = np.max(dist, axis=0)
    static_mask = max_dist < 0.15 * max_dist.max() # [N]
    return static_mask

def filter_unreasonable_motion(trajectories, vis_mask, static_threshold=0.01, line_threshold=0.05, circle_threshold=0.05):
    """
    Classify trajectories and filter out points with unreasonable motion
    
    Args:
        trajectories: Point trajectories with shape [T, N, 3]
        static_threshold: Maximum displacement to consider as static
        line_threshold: Maximum error to consider as translational motion
        circle_threshold: Maximum error to consider as rotational motion
    
    Returns:
        valid_motion: Boolean mask of shape [N] indicating points with valid motion
        motion_types: Integer array of shape [N] with motion classifications
        motion_params: List of motion parameters for each point
    """
    T, N, _ = trajectories.shape
    
    motion_types = np.zeros(N, dtype=int)
    motion_params = [None] * N
    motion_errors = np.zeros(N)
    
    static_mask = identify_static_points(trajectories, vis_mask) # [N]

    sampled_trajectories, downsample_ratios = sample_valid_trajectory(trajectories, vis_mask, 0.02)
    downsample_ratios = np.array(downsample_ratios)
    dyna_sample_ratios = downsample_ratios[~static_mask]
    print(f"Dyna sample ratios: {dyna_sample_ratios.mean()}, {dyna_sample_ratios.min()}, {dyna_sample_ratios.max()}")
    # Analyze each point's trajectory individually
    for n in tqdm(range(N), desc="Classifying trajectories"):
        if static_mask[n]:
            motion_types[n] = 1
            motion_params[n] = {}
            motion_errors[n] = 0
        elif downsample_ratios[n] < 0.1 and vis_mask[:, n].sum() < 0.2 * T: # too few valid points
            motion_types[n] = 0
            motion_params[n] = None
            motion_errors[n] = None
        else:
            motion_types[n], motion_params[n], motion_errors[n] = classify_trajectory(
                sampled_trajectories[n],
                static_threshold=static_threshold,
                line_threshold=line_threshold,
                circle_threshold=circle_threshold
            )
    
    # Points with valid motion are those classified as one of the known types
    valid_motion = motion_types > 0
    
    return valid_motion, motion_types, motion_params

def visualize_motion_types(trajectories, motion_types, motion_params, save_path=None):
    """
    Visualize trajectories colored by motion type
    
    Args:
        trajectories: Point trajectories with shape [T, N, 3]
        motion_types: Integer array of shape [N] with motion classifications
        save_path: Optional path to save the visualization
    """
    T, N, _ = trajectories.shape
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    
    # Color map for motion types
    colors = {
        0: 'purple',    # Unknown
        1: 'black',    # Static
        2: 'green',   # Translation
        3: 'red'      # Rotation
    }
    
    # Plot each trajectory
    for n in range(N):
        if motion_types[n] == 0:
            continue
        color = colors[motion_types[n]]
        if motion_types[n] == 3 and (np.abs(motion_params[n]["origin"]) > 2).any():
            color = 'yellow'
        ax.plot(trajectories[:, n, 0], trajectories[:, n, 1], trajectories[:, n, 2], 
                color=color, alpha=0.8, linewidth=1)
    
    # Mark static points
    static_points = trajectories[:, motion_types==1]
    ax.scatter(static_points[0, :, 0], static_points[0, :, 1], static_points[0, :, 2], 
               c='black', alpha=0.2, s=4)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='purple', lw=2, label='Unknown'),
        Line2D([0], [0], color='black', lw=2, label='Static'),
        Line2D([0], [0], color='green', lw=2, label='Translation'),
        Line2D([0], [0], color='red', lw=2, label='Rotation')
    ]
    ax.legend(handles=legend_elements)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trajectory Motion Type Classification')
    
    # plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    for n in range(N):
        if motion_types[n] == 0:
            color = colors[motion_types[n]]
            ax.plot(trajectories[:, n, 0], trajectories[:, n, 1], trajectories[:, n, 2], 
                    color=color, alpha=0.8, linewidth=1)
    plt.savefig(save_path.replace('.png', '_all.png'))
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
        # Select a few examples of each motion type
        static_indices = np.where(motion_types == 1)[0][:min(100, np.sum(motion_types == 1))]
        transl_indices = np.where(motion_types == 2)[0][:min(100, np.sum(motion_types == 2))]
        rot_indices = np.where(motion_types == 3)[0][:min(100, np.sum(motion_types == 3))]
        indices = np.concatenate([static_indices, transl_indices, rot_indices])
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color map for motion types
    colors = {
        1: 'black',    # Static
        2: 'green',   # Translation
        3: 'red'      # Rotation
    }
    
    for idx in indices:
        motion_type = motion_types[idx]
        traj = trajectories[:, idx, :]
        color = colors[motion_type]
        
        # Plot original trajectory
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                color=color, alpha=0.8, linewidth=2, label=f'Point {idx}')
        
        # Plot fitted model
        if motion_type == 1:  # Static
            mean_pos = traj.mean(axis=0)
            ax.scatter([mean_pos[0]], [mean_pos[1]], [mean_pos[2]], 
                       color=color, s=100, marker='o', edgecolors='black')
            
        elif motion_type == 2:  # Translation (Line)
            # Plot fitted line
            start_point = motion_params[idx]['start_point']
            end_point = motion_params[idx]['end_point']
            line_points = np.stack([start_point, end_point], axis=0)
            ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 
                   color=color, linestyle='--', linewidth=1)
            
        elif motion_type == 3:  # Rotation (Circle)
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
        print(f"Model visualization saved to {save_path}")
    else:
        plt.savefig('fitted_models_visualization.png')
        print("Model visualization saved to fitted_models_visualization.png")
    
    plt.close()

def print_motion_statistics(motion_types):
    """
    Print statistics about detected motion types
    
    Args:
        motion_types: Integer array of shape [N] with motion classifications
    """
    N = len(motion_types)
    
    n_unknown = np.sum(motion_types == 0)
    n_static = np.sum(motion_types == 1)
    n_translation = np.sum(motion_types == 2)
    n_rotation = np.sum(motion_types == 3)
    
    print(f"Motion Type Statistics:")
    print(f"  Static points: {n_static} ({n_static/N*100:.1f}%)")
    print(f"  Translational points: {n_translation} ({n_translation/N*100:.1f}%)")
    print(f"  Rotational points: {n_rotation} ({n_rotation/N*100:.1f}%)")
    print(f"  Unknown motion: {n_unknown} ({n_unknown/N*100:.1f}%)")
    print(f"  Total valid motion: {N-n_unknown} ({(N-n_unknown)/N*100:.1f}%)") 