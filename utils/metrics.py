
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import load_ply, load_obj
from pytorch3d.structures import Meshes
import torch
import open3d as o3d
import numpy as np
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
import json
from pytorch_lightning import seed_everything
from itertools import permutations
from piq import LPIPS
from piq import ssim as ssim_func
import os
from scipy.optimize import linear_sum_assignment


lpips = LPIPS()

def mse(img1, img2):
    return (((img1 - img2)) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    return 20 * torch.log10(1.0 / torch.sqrt(mse(img1, img2)))


def load_mesh(path):
    if path.endswith('.ply'):
        verts, faces = load_ply(path)
    elif path.endswith('.obj'):
        obj = load_obj(path)
        verts = obj[0]
        faces = obj[1].verts_idx
    return verts, faces


def combine_pred_mesh(paths, exp_path):
    recon_mesh = o3d.geometry.TriangleMesh()
    for path in paths:
        mesh = o3d.io.read_triangle_mesh(path)
        recon_mesh += mesh
    o3d.io.write_triangle_mesh(exp_path, recon_mesh)


def compute_chamfer(recon_pts, gt_pts):
	with torch.no_grad():
		recon_pts = recon_pts.cuda()
		gt_pts = gt_pts.cuda()
		dist,_ = chamfer_distance(recon_pts, gt_pts, batch_reduction=None, single_directional=False)
		dist = dist.item()
	return dist


def compute_recon_error(recon_path, gt_path, n_samples=10000, vis=False):
    verts, faces = load_mesh(recon_path)
    recon_mesh = Meshes(verts=[verts], faces=[faces])
    verts, faces = load_mesh(gt_path)
    gt_mesh = Meshes(verts=[verts], faces=[faces])

    gt_pts = sample_points_from_meshes(gt_mesh, num_samples=n_samples)
    recon_pts = sample_points_from_meshes(recon_mesh, num_samples=n_samples)


    if vis:
        pts = gt_pts.clone().detach().squeeze().numpy()
        gt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        o3d.io.write_point_cloud("gt_points.ply", gt_pcd)
        pts = recon_pts.clone().detach().squeeze().numpy()
        recon_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        o3d.io.write_point_cloud("recon_points.ply", recon_pcd)

    return compute_chamfer(recon_pts, gt_pts)


def find_eval_perm_cd(gt_path, pred_path, num_d_joints):
    # find the best permutation of the dynamic joints by minimizing the average chamfer distance
    pred_d_ply_list = [f'{pred_path}/part_{id}.ply' for id in range(1, num_d_joints + 1)]
    gt_d_ply_list = [f'{gt_path}/part_{id}.ply' for id in range(1, num_d_joints + 1)]

    results = []
    for gt_d_ply in gt_d_ply_list:
        for pred_d_ply in pred_d_ply_list:
            try:
                results.append(compute_recon_error(pred_d_ply, gt_d_ply, n_samples=10000, vis=False) * 100)
            except:
                results.append(100)
    results = np.array(results).reshape(num_d_joints, num_d_joints)
    idx_g, idx_p = linear_sum_assignment(results)
    if len(idx_p) == num_d_joints:
        result = results[idx_g, idx_p]
        perm = [idx_p[i] for i in range(num_d_joints)]
    else: # the number of dynamic joints in pred is less than the number of dynamic joints in gt
        result = results[idx_g, idx_p]
        perm = [idx_p[i] for i in range(len(idx_p))]
        result = np.concatenate([result, np.ones(num_d_joints - len(idx_p)) * 100])
    return result, perm


def eval_CD(gt_path, pred_path, num_d_joints, n_trials=1):
    cd_s, cd_w = 0., 0.
    cd_d = [0.] * num_d_joints
    idx = list(range(1, num_d_joints + 1))
    idx_perm = None
    if num_d_joints == 1:
        perm = [0]
        idx_perm = [1]
    for seed in range(n_trials):
        seed_everything(seed)
        pred_w_ply = f'{pred_path}/whole_mesh.ply'
        pred_s_ply = f'{pred_path}/part_0.ply'
        gt_w_ply = f'{gt_path}/whole_mesh.ply'
        gt_s_ply = f'{gt_path}/part_0.ply'
        gt_d_ply_list = [f'{gt_path}/part_{id}.ply' for id in idx]
        try:
            cd_s += compute_recon_error(pred_s_ply, gt_s_ply, n_samples=10000, vis=False) * 100
            cd_w += compute_recon_error(pred_w_ply, gt_w_ply, n_samples=10000, vis=False) * 100
        except:
            cd_s += 100
            cd_w += 100
        if idx_perm is None:
            # For multi-part objects
            # find the best permutation of the dynamic joints by minimizing the average chamfer distance
            cd_d_list, perm = find_eval_perm_cd(gt_path, pred_path, num_d_joints)
            idx_perm = [idx[i] for i in perm]
            for i in range(num_d_joints):
                cd_d[i] += cd_d_list[i]
        else:
            pred_d_ply_list = [f'{pred_path}/part_{id}.ply' for id in idx_perm]
            for i in range(num_d_joints):
                try:
                    cd_d[i] += compute_recon_error(pred_d_ply_list[i], gt_d_ply_list[i], n_samples=10000, vis=False) * 100
                except:
                    cd_d[i] += 100
    cd_s /= n_trials
    cd_w /= n_trials
    # print(f'CD_static {cd_s:.4f}', end=', ')
    for i in range(num_d_joints):
        cd_d[i] /= n_trials
        # print(f'CD_dynamic_{i} {cd_d[i]:.4f}', end=', ') 
    # print(f'CD_whole {cd_w:.4f}')
    return cd_s, cd_d, cd_w, perm


def interpret_transforms(base_R, base_t, R, t, joint_type='revolute'):
    """
    base_R, base_t, R, t are all from canonical to world
    rewrite the transformation = global transformation (base_R, base_t) {R' part + t'} --> s.t. R' and t' happens in canonical space
    R', t':
    - revolute: R'p + t' = R'(p - a) + a, R' --> axis-theta representation; axis goes through a = (I - R')^{-1}t'
    - prismatic: R' = I, t' = l * direction
    """
    R = np.matmul(base_R.T, R)
    t = np.matmul(base_R.T, (t - base_t).reshape(3, 1)).reshape(-1)

    if joint_type == 'revolute':
        rotvec = Rotation.from_matrix(R).as_rotvec()
        theta = np.linalg.norm(rotvec, axis=-1)
        direction = rotvec / max(theta, (theta < 1e-8))
        try:
            origin = np.matmul(np.linalg.inv(np.eye(3) - R), t.reshape(3, 1)).reshape(-1)
        except:   # TO DO find the best solution
            origin = np.zeros(3)
        origin += direction * np.dot(direction, -origin)
        joint_info = {'origin': origin,
                      'direction': direction,
                      'theta': np.rad2deg(theta),
                      'rotation': R, 'translation': t}

    elif joint_type == 'prismatic':
        theta = np.linalg.norm(t)
        direction = t / max(theta, (theta < 1e-8))
        joint_info = {'direction': direction, 'origin': np.zeros(3), 'theta': theta,
                      'rotation': R, 'translation': t}

    return joint_info, R, t


def line_distance(a_o, a_d, b_o, b_d):
    normal = np.cross(a_d, b_d)
    normal_length = np.linalg.norm(normal)
    if normal_length < 1e-6:   # parallel
        return np.linalg.norm(np.cross(b_o - a_o, a_d))
    else:
        return np.abs(np.dot(normal, a_o - b_o)) / normal_length


def eval_axis_and_state(axis_a, axis_b, joint_type='r'):
    a_d, b_d = axis_a['direction'], axis_b['direction']
    angle = np.rad2deg(np.arccos(np.dot(a_d, b_d) / np.linalg.norm(a_d) / np.linalg.norm(b_d)))
    angle = min(angle, 180 - angle)

    if joint_type == 'r':
        a_o, b_o = axis_a['origin'], axis_b['origin']
        distance = line_distance(a_o, a_d, b_o, b_d)
    elif joint_type == 'p':
        distance = 0
    else:
        raise ValueError(f'Unknown joint type {joint_type}')

    return angle, distance * 100 # deg, cm


def find_eval_perm_axis(pred_joint_list, gt_joint_list):
    # find the best permutation of the dynamic joints by minimizing the average chamfer distance
    num_d_joints = len(gt_joint_list)
    results = []
    for gt_joint in gt_joint_list:
        for pred_joint in pred_joint_list:
            if pred_joint["joint_type"] != gt_joint["joint_type"]:
                angle, distance = 90, 100 # deg, cm
            else:
                angle, distance = eval_axis_and_state(pred_joint, gt_joint, gt_joint["joint_type"])
            results.append([angle, distance])
    results = np.array(results).reshape(num_d_joints, len(pred_joint_list), 2)
    score = results[..., 0] + results[..., 1]
    idx_g, idx_p = linear_sum_assignment(score)
    if len(idx_p) == num_d_joints:
        result = results[idx_g, idx_p]
        perm = [idx_p[i] for i in range(num_d_joints)]
    else: # the number of dynamic joints in pred is less than the number of dynamic joints in gt
        result = results[idx_g, idx_p]
        perm = [idx_p[i] for i in range(len(idx_p))]
        res = np.ones((num_d_joints - len(idx_p), 2)) * 100
        res[:, 0] = 90
        result = np.concatenate([result, res], axis=0)
    return result, perm


def eval_axis_and_state_all(pred_joint_list, gt_joint_list, perm=None):
    if perm is not None:
        pred_joint_list = [pred_joint_list[i] for i in perm]
        num_d_joints = len(pred_joint_list)
        results = []
        for i in range(num_d_joints):
            pred_joint = pred_joint_list[i]
            gt_joint = gt_joint_list[i]
            angle, distance = eval_axis_and_state(pred_joint, gt_joint, gt_joint["joint_type"])
            results.append((angle, distance))
    else:
        results, perm = find_eval_perm_axis(pred_joint_list, gt_joint_list)
    return results, perm
    

def geodesic_distance(pred_R, gt_R):
    '''
    q is the output from the network (rotation from t=0.5 to t=1)
    gt_R is the GT rotation from t=0 to t=1
    '''
    pred_R, gt_R = pred_R.cpu(), gt_R.cpu()
    R_diff = torch.matmul(pred_R, gt_R.T)
    cos_angle = torch.clip((torch.trace(R_diff) - 1.0) * 0.5, min=-1., max=1.)
    angle = torch.rad2deg(torch.arccos(cos_angle)) 
    return angle


def axis_metrics(motion, gt):
    # pred axis
    pred_axis_d = motion['axis_d'].cpu().squeeze(0)
    pred_axis_o = motion['axis_o'].cpu().squeeze(0)
    # gt axis
    gt_axis_d = gt['axis_d']
    gt_axis_o = gt['axis_o']
    # angular difference between two vectors
    cos_theta = torch.dot(pred_axis_d, gt_axis_d) / (torch.norm(pred_axis_d) * torch.norm(gt_axis_d))
    ang_err = torch.rad2deg(torch.acos(torch.abs(cos_theta)))
    # positonal difference between two axis lines
    w = gt_axis_o - pred_axis_o
    cross = torch.cross(pred_axis_d, gt_axis_d)
    if (cross == torch.zeros(3)).sum().item() == 3:
        pos_err = torch.tensor(0)
    else:
        pos_err = torch.abs(torch.sum(w * cross)) / torch.norm(cross)
    return ang_err, pos_err


def translational_error(motion, gt):
    dist_half = motion['dist'].cpu()
    dist = dist_half * 2.
    gt_dist = gt['dist']

    axis_d = F.normalize(motion['axis_d'].cpu().squeeze(0), p=2, dim=0)
    gt_axis_d = F.normalize(gt['axis_d'].cpu(), p=2, dim=0)

    err = torch.sqrt(((dist * axis_d - gt_dist * gt_axis_d) ** 2).sum())
    return err


joint_type_dict = {
    'r': 'hinge',
    'p': 'slider',
}


def read_gt(gt_path):
    with open(gt_path, 'r') as f:
        info = json.load(f)
    ret_list = []
    if info and 'joint_id' in info[0] and 'type' in info[0]:
        R_coord = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
        for entry in info:
            if entry['type'] == 'revolute':
                joint_type = 'r'
            elif entry['type'] == 'prismatic':
                joint_type = 'p'
            else:
                continue
            axis_o = np.array(entry.get('axis_point', [0, 0, 0]), dtype=np.float32)
            axis_d = np.array(entry['axis_dir'], dtype=np.float32)
            axis_o = np.matmul(R_coord, axis_o)
            axis_d = np.matmul(R_coord, axis_d)
            ret_list.append({
                'origin': axis_o,
                'direction': axis_d,
                'joint_type': joint_type,
                'idx': entry['joint_id'],
            })
        return ret_list
    if 'v2a' in gt_path:
        R_coord = Rotation.from_euler('xyz', [90, 0, -90], degrees=True).as_matrix()
        for entry in info:
            if entry['joint'] == 'hinge':
                joint_type = 'r'
            elif entry['joint'] == 'slider':
                joint_type = 'p'
            else:
                continue
            axis_o = np.array(entry['jointData']['axis']['origin'])
            axis_d = np.array(entry['jointData']['axis']['direction'])
            axis_o = np.matmul(R_coord, axis_o)
            axis_d = np.matmul(R_coord, axis_d)
            ret_list.append({'origin': axis_o, 'direction': axis_d, "joint_type": joint_type})
    else:
        R_coord = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
        base_ids = []
        for entry in info:
            if entry['parent'] == -1:
                base_ids.append(entry['id'])
        for entry in info:
            if entry['name'] in ['knob', 'tray', 'control_button', 'button'] or entry['parent'] not in base_ids:
                continue
            if entry['name'] == 'handle' and '100481' not in gt_path:
                continue
            if entry['joint'] == 'hinge':
                joint_type = 'r'
            elif entry['joint'] == 'slider':
                joint_type = 'p'
            else:
                continue
            axis_o = np.array(entry['jointData']['axis']['origin'])
            axis_d = np.array(entry['jointData']['axis']['direction'])
            axis_o = np.matmul(R_coord, axis_o)
            axis_d = np.matmul(R_coord, axis_d)
            ret_list.append({'origin': axis_o, 'direction': axis_d, "joint_type": joint_type, "idx": entry['id']})
    return ret_list


def read_joint_infos(joint_infos_path):
    with open(joint_infos_path, 'r') as f:
        info = json.load(f)
    ret_list = []
    for entry in info:
        if entry['joint_type'] != 's':
            ret_list.append({
                'direction': np.array(entry['direction'], dtype=np.float32), 
                'origin': np.array(entry['origin'], dtype=np.float32), 
                "joint_type": entry['joint_type']
            })
    return ret_list


def read_joint_priori(joint_priori_path):
    with open(joint_priori_path, 'r') as f:
        info = json.load(f)
    ret_list = []
    for entry in info:
        if entry['joint'] == 'hinge':
            joint_type = 'r'
        elif entry['joint'] == 'slider':
            joint_type = 'p'
        else:
            continue
        ret_list.append({'joint_type': joint_type})
    return ret_list


def load_joint_infos(joint_infos_path):
    with open(joint_infos_path, 'r') as f:
        info = json.load(f)
    ret_list = []
    for entry in info:
        if entry['joint'] == 'hinge':
            joint_type = 'r'
        elif entry['joint'] == 'slider':
            joint_type = 'p'
        else:
            continue
        axis_o = np.array(entry['jointData']['axis']['origin'])
        axis_d = np.array(entry['jointData']['axis']['direction'])
        ret_list.append({'origin': axis_o, 'direction': axis_d, 
                         "joint_type": joint_type,
                         'center': entry['center']})
    return ret_list


def export_joint_info_json(pred_joint_list, mesh_files, joint_limit, exp_dir):
    meta_info = []
    for i, joint_info in enumerate(pred_joint_list):
        if i == 0:
            entry = {
            "id": i,
            "parent": -1,
            "name": "root",
            "joint": 'heavy',
            "jointData": {},
            "center": [0, 0, 0],
            "visuals": [
                mesh_files[i]
            ]
        }
        else:
            entry = {
                "id": i,
                "parent": 0,
                "name": f"joint_{i}",
                "joint": joint_type_dict[joint_info["joint_type"]],
                "jointData": {
                    "axis": {
                        "origin": joint_info['origin'].tolist(),
                        "direction": joint_info['direction'].tolist()
                    },
                    "limit": {
                        "lower": joint_limit[0][i].item(),
                        "upper": joint_limit[1][i].item()
                    }
                },
                "center": joint_info['center'].tolist(),
                "visuals": [
                    mesh_files[i]
                ]
            }
        meta_info.append(entry)
    with open(os.path.join(exp_dir, 'joint_info.json'), 'w') as f:
        json.dump(meta_info, f, indent=4)
