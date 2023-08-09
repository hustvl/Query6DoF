import os
import logging
import math
import cv2
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import _pickle as cPickle
from tqdm import tqdm
from mmengine import Registry
import torch
from torch.optim import SGD,AdamW
from torch.optim.lr_scheduler import MultiStepLR,CosineAnnealingLR
import copy

def build_scheduler(cfg, registry, optimizer: torch.optim.Optimizer):
    sche=registry.get(cfg.type)
    sche_cfg=copy.deepcopy(cfg)
    del sche_cfg['type']
    sche_cfg['optimizer']=optimizer
    return sche(**sche_cfg)

SCHEDULER_REGISTRY = Registry("SCHEDULER_REGISTRY",build_func=build_scheduler)
SCHEDULER_REGISTRY.register_module(module=MultiStepLR)
SCHEDULER_REGISTRY.register_module(module=CosineAnnealingLR)


def build_optimizer(cfg, registry, parameters):
    opt_cfg=copy.deepcopy(cfg)
    opt=registry.get(cfg.type)
    del opt_cfg['type']
    opt_cfg['params']=parameters
    return opt(**opt_cfg)

OPTIMIZER_REGISTRY = Registry("OPTIMIZER_REGISTRY",build_func=build_optimizer)
OPTIMIZER_REGISTRY.register_module(module=AdamW)
OPTIMIZER_REGISTRY.register_module(module=SGD)


# point cloud based data augmentation
# augment based on bounding box
def defor_3D_bb(pc, R, t, s, nocs, model, sym=None, aug_bb=None):
    # pc  n x 3, here s must  be the original s
    pc_reproj = torch.mm(R.T, (pc - t.view(1, 3)).T).T  # nn x 3
    if sym[0] == 1:  # y axis symmetry
        ex = aug_bb[0]
        ey = aug_bb[1]
        ez = aug_bb[2]

        exz = (ex + ez) / 2
        pc_reproj[:, (0, 2)] = pc_reproj[:, (0, 2)] * exz
        pc_reproj[:, 1] = pc_reproj[:, 1] * ey
        nocs_scale_aug = torch.norm(torch.tensor([s[0] * exz, s[1] * ey, s[2] * exz])) / torch.norm(s)
        s[0] = s[0] * exz
        s[1] = s[1] * ey
        s[2] = s[2] * exz
        nocs[:, 0] = nocs[:, 0] * exz / nocs_scale_aug
        nocs[:, 1] = nocs[:, 1] * ey / nocs_scale_aug
        nocs[:, 2] = nocs[:, 2] * exz / nocs_scale_aug
        model[:, 0] = model[:, 0] * exz / nocs_scale_aug
        model[:, 1] = model[:, 1] * ey / nocs_scale_aug
        model[:, 2] = model[:, 2] * exz / nocs_scale_aug
        pc_new = torch.mm(R, pc_reproj.T) + t.view(3, 1)
        pc_new = pc_new.T
    else:
        ex = aug_bb[0]
        ey = aug_bb[1]
        ez = aug_bb[2]
        nocs_scale_aug = torch.norm(torch.tensor([s[0] * ex, s[1] * ey, s[2] * ez])) / torch.norm(s)
        pc_reproj[:, 0] = pc_reproj[:, 0] * ex
        pc_reproj[:, 1] = pc_reproj[:, 1] * ey
        pc_reproj[:, 2] = pc_reproj[:, 2] * ez
        s[0] = s[0] * ex
        s[1] = s[1] * ey
        s[2] = s[2] * ez
        nocs[:, 0] = nocs[:, 0] * ex / nocs_scale_aug
        nocs[:, 1] = nocs[:, 1] * ey / nocs_scale_aug
        nocs[:, 2] = nocs[:, 2] * ez / nocs_scale_aug
        model[:, 0] = model[:, 0] * ex / nocs_scale_aug
        model[:, 1] = model[:, 1] * ey / nocs_scale_aug
        model[:, 2] = model[:, 2] * ez / nocs_scale_aug
        pc_new = torch.mm(R, pc_reproj.T) + t.view(3, 1)
        pc_new = pc_new.T
    return pc_new, s, nocs, model


def defor_3D_bc(pc, R, t, s, model_point, nocs_scale, nocs):
    # resize box cage along y axis, the size s is modified
    ey_up = torch.rand(1, device=pc.device) * (1.2 - 0.8) + 0.8
    ey_down = torch.rand(1,  device=pc.device) * (1.2 - 0.8) + 0.8
    # for each point, resize its x and z linealy
    pc_reproj = torch.mm(R.T, (pc - t.view(1, 3)).T).T  # nn x 3
    per_point_resize = (pc_reproj[:, 1] + s[1] / 2) / s[1] * (ey_up - ey_down) + ey_down
    pc_reproj[:, 0] = pc_reproj[:, 0] * per_point_resize
    pc_reproj[:, 2] = pc_reproj[:, 2] * per_point_resize
    pc_new = torch.mm(R, pc_reproj.T) + t.view(3, 1)
    pc_new = pc_new.T

    norm_s = s / torch.norm(s)
    model_point_resize =  (model_point[:, 1] + norm_s[1] / 2) / norm_s[1] * (ey_up - ey_down) + ey_down
    model_point[:, 0] = model_point[:, 0] * model_point_resize
    model_point[:, 2] = model_point[:, 2] * model_point_resize

    lx = 2 * max(max(model_point[:, 0]), -min(model_point[:, 0]))
    ly = max(model_point[:, 1]) - min(model_point[:, 1])
    lz = max(model_point[:, 2]) - min(model_point[:, 2])

    lx_t = lx * torch.norm(s)
    ly_t = ly * torch.norm(s)
    lz_t = lz * torch.norm(s)
    size_new = torch.tensor([lx_t, ly_t, lz_t], device=pc.device)

    nocs_scale_aug = torch.norm(torch.tensor([lx, ly, lz]))
    model_point = model_point / nocs_scale_aug

    nocs_resize = (nocs[:, 1] + norm_s[1] / 2) / norm_s[1] * (ey_up - ey_down) + ey_down
    nocs[:, 0] = nocs[:, 0] * nocs_resize
    nocs[:, 2] = nocs[:, 2] * nocs_resize
    nocs = nocs / nocs_scale_aug

    return pc_new, size_new, model_point, nocs


# point cloud based data augmentation
# augment based on bounding box
def deform_non_linear(pc, R, t, s, nocs, model_point, axis=0):
    # pc  n x 3, here s must  be the original s
    assert axis in [0, 1]
    r_max = torch.rand(1, device=pc.device) * 0.2 + 1.1
    r_min = -torch.rand(1, device=pc.device) * 0.2 + 0.9
    # for each point, resize its x and z
    pc_reproj = torch.mm(R.T, (pc - t.view(1, 3)).T).T  # nn x 3
    per_point_resize = r_min + 4 * (pc_reproj[:, axis] * pc_reproj[:, axis]) / (s ** 2) * (r_max - r_min)
    pc_reproj[:, axis] = pc_reproj[:, axis] * per_point_resize

    r=torch.rand(1,device=pc.device)*0.4+0.8
    for i in range(3):
        if i==axis:
            continue
        pc_reproj[:,i]=pc_reproj[:,i]*r

    pc_new = torch.mm(R, pc_reproj.T) + t.view(3, 1)
    pc_new = pc_new.T

    #norm_s = s / torch.norm(s)
    model_point_resize = r_min + 4 * (model_point[:, axis] * model_point[:, axis]) / (s ** 2) * (r_max - r_min)
    model_point[:, axis] = model_point[:, axis] * model_point_resize

    for i in range(3):
        if i==axis:
            continue
        model_point[:,i]=model_point[:,i]*r

    lx = 2 * max(max(model_point[:, 0]), -min(model_point[:, 0]))
    ly = max(model_point[:, 1]) - min(model_point[:, 1])
    lz = max(model_point[:, 2]) - min(model_point[:, 2])

    lx_t = lx * torch.norm(s)
    ly_t = ly * torch.norm(s)
    lz_t = lz * torch.norm(s)
    size_new = torch.tensor([lx_t, ly_t, lz_t], device=pc.device)
    size_new=torch.norm(size_new)

    nocs_scale_aug = torch.norm(torch.tensor([lx, ly, lz]))
    model_point = model_point / nocs_scale_aug


    nocs_resize = r_min + 4 * (nocs[:, axis] * nocs[:, axis]) / (s ** 2) * (r_max - r_min)
    nocs[:, axis] = nocs[:, axis] * nocs_resize

    for i in range(3):
        if i==axis:
            continue
        nocs[:,i]=nocs[:,i]*r

    nocs = nocs / nocs_scale_aug
    return pc_new, size_new, model_point, nocs


def defor_3D_pc(pc, r):
    points_defor = torch.randn(pc.shape).to(pc.device)
    pc = pc + points_defor * r
    return pc


# point cloud based data augmentation
# random rotation and translation
def defor_3D_rt(pc, R, t, aug_rt_t, aug_rt_r):
    #  add_t
    dx = aug_rt_t[0]
    dy = aug_rt_t[1]
    dz = aug_rt_t[2]

    pc[:, 0] = pc[:, 0] + dx
    pc[:, 1] = pc[:, 1] + dy
    pc[:, 2] = pc[:, 2] + dz
    t[0] = t[0] + dx
    t[1] = t[1] + dy
    t[2] = t[2] + dz

    # add r
    '''
    Rm = get_rotation(np.random.uniform(-a, a), np.random.uniform(-a, a), np.random.uniform(-a, a))
    Rm_tensor = torch.tensor(Rm, device=pc.device)
    pc_new = torch.mm(Rm_tensor, pc.T).T
    pc = pc_new
    R_new = torch.mm(Rm_tensor, R)
    R = R_new
    '''
    '''
    x_rot = torch.rand(1, dtype=torch.float32, device=pc.device) * 2 * a - a
    y_rot = torch.rand(1, dtype=torch.float32, device=pc.device) * 2 * a - a
    z_rot = torch.rand(1, dtype=torch.float32, device=pc.device) * 2 * a - a
    Rm = get_rotation_torch(x_rot, y_rot, z_rot)
    '''
    Rm = aug_rt_r
    pc_new = torch.mm(Rm, pc.T).T
    pc = pc_new
    R_new = torch.mm(Rm, R)
    R = R_new
    T_new = torch.mm(Rm, t.view(3, 1))
    t = T_new

    return pc, R, t



def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, 'model', filename))

def judge_nan(tensor):
    t=torch.isnan(tensor).float().sum().item()
    return t==0

def draw_two_coords(image,coord1,coord2,s,RT,cam_K):
    assert len(coord1.shape)==2
    if coord1 is not None and coord2 is not None:
        image=draw_lines(image,coord1*s,coord2*s,RT,RT,cam_K)
    if coord1 is not None:
        image=draw_3d_points(image,coord1*s,RT,cam_K,color=(0,0,255))
    if coord2 is not None:
        image=draw_3d_points(image,coord2*s,RT,cam_K,color=(255,0,0)) 
    return image



def draw_points_coords(image,coords,points,s,RT,cam_K,coord_color=(255,0,0)):
    assert len(coords.shape)==2
    if points is not None and coords is not None:
        image=draw_lines(image,points,coords*s,np.identity(4,dtype=np.float32),RT,cam_K)
    if points is not None:
        image=draw_3d_points(image,points,np.identity(4,dtype=np.float32),cam_K,color=(0,0,255))
    if coords is not None:
        image=draw_3d_points(image,coords*s,RT,cam_K,color=coord_color)    

    return image

def draw_lines(image,p1,p2,RT1,RT2,cam_K):
    kp_2d1,_=get_project(p1,RT1,cam_K)
    kp_2d2,_=get_project(p2,RT2,cam_K)
    for p1,p2 in zip(kp_2d1,kp_2d2):
        cv2.line(image,tuple(p1),tuple(p2),(0,255,0),1)
    return image

def draw_3d_points(image,points,RT,cam_K,color=(0,0,255)):
    kp_2d,_=get_project(points,RT,cam_K)
    for point in kp_2d:
        cv2.circle(image,tuple(point),1,color,-1)
    return image

def get_project(kp3d,gt_pose,intrinsics):
    kp3d=kp3d.reshape(-1,1,3)
    depth=cv2.perspectiveTransform(kp3d,gt_pose)[:,0,2]
    kp3d = np.ascontiguousarray(kp3d.astype(np.float64))
    intrinsics = intrinsics.astype(np.float64)
    if gt_pose.shape == (4, 4):
        gt_pose = gt_pose[:3, :4]
    tvec=gt_pose[:,3]
    rotation=gt_pose[:,:3]
    rotation=cv2.Rodrigues(rotation)[0]
    dist_coeffs = np.zeros(shape=[8, 1], dtype='float64')
    kp2d=cv2.projectPoints(kp3d,rotation,tvec,intrinsics,dist_coeffs)[0][:,0,:].astype(np.uint16)
    return kp2d,depth


def setup_logger(logger_name, log_file, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(fileHandler)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    return logger


def estimateSimilarityUmeyama(SourceHom, TargetHom, wo_scale=False):
    # Copy of original paper is at: http://web.stanford.edu/class/cs273/refs/umeyama.pdf
    SourceCentroid = np.mean(SourceHom[:3, :], axis=1)
    TargetCentroid = np.mean(TargetHom[:3, :], axis=1)
    nPoints = SourceHom.shape[1]
    CenteredSource = SourceHom[:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()
    CenteredTarget = TargetHom[:3, :] - np.tile(TargetCentroid, (nPoints, 1)).transpose()
    CovMatrix = np.matmul(CenteredTarget, np.transpose(CenteredSource)) / nPoints
    if np.isnan(CovMatrix).any():
        print('nPoints:', nPoints)
        print(SourceHom.shape)
        print(TargetHom.shape)
        raise RuntimeError('There are NANs in the input.')

    U, D, Vh = np.linalg.svd(CovMatrix, full_matrices=True)
    d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]
    # rotation
    Rotation = np.matmul(U, Vh)
    # scale
    varP = np.var(SourceHom[:3, :], axis=1).sum()
    Scale = 1 / varP * np.sum(D)
    # translation
    Translation = TargetHom[:3, :].mean(axis=1) - SourceHom[:3, :].mean(axis=1).dot(Scale*Rotation.T)
    # transformation matrix
    OutTransform = np.identity(4)
    if not wo_scale:
        OutTransform[:3, :3] = Scale * Rotation
    else:
        OutTransform[:3, :3] = Rotation
    OutTransform[:3, 3] = Translation

    return Scale, Rotation, Translation, OutTransform


def estimateSimilarityTransform(source: np.array, target: np.array, verbose=False):
    """ Add RANSAC algorithm to account for outliers.

    """
    assert source.shape[0] == target.shape[0], 'Source and Target must have same number of points.'
    SourceHom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))
    TargetHom = np.transpose(np.hstack([target, np.ones([target.shape[0], 1])]))
    # Auto-parameter selection based on source heuristics
    # Assume source is object model or gt nocs map, which is of high quality
    SourceCentroid = np.mean(SourceHom[:3, :], axis=1)
    nPoints = SourceHom.shape[1]
    CenteredSource = SourceHom[:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()
    SourceDiameter = 2 * np.amax(np.linalg.norm(CenteredSource, axis=0))
    InlierT = SourceDiameter / 10.0  # 0.1 of source diameter
    maxIter = 128
    confidence = 0.99

    if verbose:
        print('Inlier threshold: ', InlierT)
        print('Max number of iterations: ', maxIter)

    BestInlierRatio = 0
    BestInlierIdx = np.arange(nPoints)
    for i in range(0, maxIter):
        # Pick 5 random (but corresponding) points from source and target
        RandIdx = np.random.randint(nPoints, size=5)
        Scale, _, _, OutTransform = estimateSimilarityUmeyama(SourceHom[:, RandIdx], TargetHom[:, RandIdx])
        PassThreshold = Scale * InlierT    # propagate inlier threshold to target scale
        Diff = TargetHom - np.matmul(OutTransform, SourceHom)
        ResidualVec = np.linalg.norm(Diff[:3, :], axis=0)
        InlierIdx = np.where(ResidualVec < PassThreshold)[0]
        nInliers = InlierIdx.shape[0]
        InlierRatio = nInliers / nPoints
        # update best hypothesis
        if InlierRatio > BestInlierRatio:
            BestInlierRatio = InlierRatio
            BestInlierIdx = InlierIdx
        if verbose:
            print('Iteration: ', i)
            print('Inlier ratio: ', BestInlierRatio)
        # early break
        if (1 - (1 - BestInlierRatio ** 5) ** i) > confidence:
            break

    if(BestInlierRatio < 0.1):
        print('[ WARN ] - Something is wrong. Small BestInlierRatio: ', BestInlierRatio)
        return None, None, None, None

    SourceInliersHom = SourceHom[:, BestInlierIdx]
    TargetInliersHom = TargetHom[:, BestInlierIdx]
    Scale, Rotation, Translation, OutTransform = estimateSimilarityUmeyama(SourceInliersHom, TargetInliersHom)

    if verbose:
        print('BestInlierRatio:', BestInlierRatio)
        print('Rotation:\n', Rotation)
        print('Translation:\n', Translation)
        print('Scale:', Scale)

    return Scale, Rotation, Translation, OutTransform


def cm_degree_5_metric(pose_pred, pose_target):
    translation_distance = np.linalg.norm(pose_pred[:3, 3] - pose_target[:3, 3]) * 100
    rotation_diff = np.dot(pose_pred[:3, :3], pose_target[:3, :3].T)
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
    return (translation_distance,angular_distance)
    


def load_obj(path_to_file):
    """ Load obj file.

    Args:
        path_to_file: path

    Returns:
        vertices: ndarray
        faces: ndarray, index of triangle vertices

    """
    vertices = []
    faces = []
    with open(path_to_file, 'r') as f:
        for line in f:
            if line[:2] == 'v ':
                vertex = line[2:].strip().split(' ')
                vertex = [float(xyz) for xyz in vertex]
                vertices.append(vertex)
            elif line[0] == 'f':
                face = line[1:].replace('//', '/').strip().split(' ')
                face = [int(idx.split('/')[0])-1 for idx in face]
                faces.append(face)
            else:
                continue
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    return vertices, faces


def create_sphere():
    # 642 verts, 1280 faces,
    verts, faces = load_obj('assets/sphere_mesh_template.obj')
    return verts, faces


def random_point(face_vertices):
    """ Sampling point using Barycentric coordiante.

    """
    r1, r2 = np.random.random(2)
    sqrt_r1 = np.sqrt(r1)
    point = (1 - sqrt_r1) * face_vertices[0, :] + \
        sqrt_r1 * (1 - r2) * face_vertices[1, :] + \
        sqrt_r1 * r2 * face_vertices[2, :]

    return point


def pairwise_distance(A, B):
    """ Compute pairwise distance of two point clouds.point

    Args:
        A: n x 3 numpy array
        B: m x 3 numpy array

    Return:
        C: n x m numpy array

    """
    diff = A[:, :, None] - B[:, :, None].T
    C = np.sqrt(np.sum(diff**2, axis=1))

    return C


def uniform_sample(vertices, faces, n_samples, with_normal=False):
    """ Sampling points according to the area of mesh surface.

    """
    sampled_points = np.zeros((n_samples, 3), dtype=float)
    normals = np.zeros((n_samples, 3), dtype=float)
    faces = vertices[faces]
    vec_cross = np.cross(faces[:, 1, :] - faces[:, 0, :],
                         faces[:, 2, :] - faces[:, 0, :])
    face_area = 0.5 * np.linalg.norm(vec_cross, axis=1)
    cum_area = np.cumsum(face_area)
    for i in range(n_samples):
        face_id = np.searchsorted(cum_area, np.random.random() * cum_area[-1])
        sampled_points[i] = random_point(faces[face_id, :, :])
        normals[i] = vec_cross[face_id]
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    if with_normal:
        sampled_points = np.concatenate((sampled_points, normals), axis=1)
    return sampled_points


def farthest_point_sampling(points, n_samples):
    """ Farthest point sampling.

    """
    selected_pts = np.zeros((n_samples,), dtype=int)
    dist_mat = pairwise_distance(points, points)
    # start from first point
    pt_idx = 0
    dist_to_set = dist_mat[:, pt_idx]
    for i in range(n_samples):
        selected_pts[i] = pt_idx
        dist_to_set = np.minimum(dist_to_set, dist_mat[:, pt_idx])
        pt_idx = np.argmax(dist_to_set)
    return selected_pts


def sample_points_from_mesh(path, n_pts, with_normal=False, fps=False, ratio=2):
    """ Uniformly sampling points from mesh model.

    Args:
        path: path to OBJ file.
        n_pts: int, number of points being sampled.
        with_normal: return points with normal, approximated by mesh triangle normal
        fps: whether to use fps for post-processing, default False.
        ratio: int, if use fps, sample ratio*n_pts first, then use fps to sample final output.

    Returns:
        points: n_pts x 3, n_pts x 6 if with_normal = True

    """
    vertices, faces = load_obj(path)
    if fps:
        points = uniform_sample(vertices, faces, ratio*n_pts, with_normal)
        pts_idx = farthest_point_sampling(points[:, :3], n_pts)
        points = points[pts_idx]
    else:
        points = uniform_sample(vertices, faces, n_pts, with_normal)
    return points


def load_depth(img_path):
    """ Load depth image from img_path. """
    depth_path = img_path + '_depth.png'
    depth = cv2.imread(depth_path, -1)
    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        # NOTE: RGB is actually BGR in opencv
        depth16 = 256*(depth[:, :, 1]) + depth[:, :, 2]
        depth16 = np.where(depth16==32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    return depth16


def get_bbox(bbox):
    """ Compute square image crop window. """
    y1, x1, y2, x2 = bbox
    img_width = 480
    img_length = 640
    window_size = (max(y2-y1, x2-x1) // 40 + 1) * 40
    window_size = min(window_size, 440)
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


def compute_sRT_errors(sRT1, sRT2):
    """
    Args:
        sRT1: [4, 4]. homogeneous affine transformation
        sRT2: [4, 4]. homogeneous affine transformation

    Returns:
        R_error: angle difference in degree,
        T_error: Euclidean distance
        IoU: relative scale error

    """
    try:
        assert np.array_equal(sRT1[3, :], sRT2[3, :])
        assert np.array_equal(sRT1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(sRT1[3, :], sRT2[3, :])

    s1 = np.cbrt(np.linalg.det(sRT1[:3, :3]))
    R1 = sRT1[:3, :3] / s1
    T1 = sRT1[:3, 3]
    s2 = np.cbrt(np.linalg.det(sRT2[:3, :3]))
    R2 = sRT2[:3, :3] / s2
    T2 = sRT2[:3, 3]
    R12 = R1 @ R2.transpose()
    R_error = np.arccos(np.clip((np.trace(R12)-1)/2, -1.0, 1.0)) * 180 / np.pi
    T_error = np.linalg.norm(T1 - T2)
    IoU = np.abs(s1 - s2) / s2

    return R_error, T_error, IoU


############################################################
#  Evaluation
############################################################

def get_3d_bbox(size, shift=0):
    """
    Args:
        size: [3] or scalar
        shift: [3] or scalar
    Returns:
        bbox_3d: [3, N]

    """
    bbox_3d = np.array([[+size[0] / 2, +size[1] / 2, +size[2] / 2],
                        [+size[0] / 2, +size[1] / 2, -size[2] / 2],
                        [-size[0] / 2, +size[1] / 2, +size[2] / 2],
                        [-size[0] / 2, +size[1] / 2, -size[2] / 2],
                        [+size[0] / 2, -size[1] / 2, +size[2] / 2],
                        [+size[0] / 2, -size[1] / 2, -size[2] / 2],
                        [-size[0] / 2, -size[1] / 2, +size[2] / 2],
                        [-size[0] / 2, -size[1] / 2, -size[2] / 2]]) + shift
    bbox_3d = bbox_3d.transpose()
    return bbox_3d


def transform_coordinates_3d(coordinates, sRT):
    """
    Args:
        coordinates: [3, N]
        sRT: [4, 4]

    Returns:
        new_coordinates: [3, N]

    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = sRT @ coordinates
    new_coordinates = new_coordinates[:3, :] / (new_coordinates[3, :])
    return new_coordinates


def compute_3d_IoU(sRT_1, sRT_2, size_1, size_2, class_name_1, class_name_2, handle_visibility):
    """ Computes IoU overlaps between two 3D bboxes. """
    def asymmetric_3d_iou(sRT_1, sRT_2, size_1, size_2):
        noc_cube_1 = get_3d_bbox(size_1, 0)
        bbox_3d_1 = transform_coordinates_3d(noc_cube_1, sRT_1)
        noc_cube_2 = get_3d_bbox(size_2, 0)
        bbox_3d_2 = transform_coordinates_3d(noc_cube_2, sRT_2)

        bbox_1_max = np.amax(bbox_3d_1, axis=0)
        bbox_1_min = np.amin(bbox_3d_1, axis=0)
        bbox_2_max = np.amax(bbox_3d_2, axis=0)
        bbox_2_min = np.amin(bbox_3d_2, axis=0)

        overlap_min = np.maximum(bbox_1_min, bbox_2_min)
        overlap_max = np.minimum(bbox_1_max, bbox_2_max)

        # intersections and union
        if np.amin(overlap_max - overlap_min) < 0:
            intersections = 0
        else:
            intersections = np.prod(overlap_max - overlap_min)
        union = np.prod(bbox_1_max - bbox_1_min) + np.prod(bbox_2_max - bbox_2_min) - intersections
        overlaps = intersections / union
        return overlaps

    if sRT_1 is None or sRT_2 is None:
        return -1

    if (class_name_1 in ['bottle', 'bowl', 'can'] and class_name_1 == class_name_2) or \
        (class_name_1 == 'mug' and class_name_1 == class_name_2 and handle_visibility==0):
        def y_rotation_matrix(theta):
            return np.array([[ np.cos(theta), 0, np.sin(theta), 0],
                             [ 0,             1, 0,             0],
                             [-np.sin(theta), 0, np.cos(theta), 0],
                             [ 0,             0, 0,             1]])
        n = 20
        max_iou = 0
        for i in range(n):
            rotated_RT_1 = sRT_1 @ y_rotation_matrix(2 * math.pi * i / float(n))
            max_iou = max(max_iou, asymmetric_3d_iou(rotated_RT_1, sRT_2, size_1, size_2))
    else:
        max_iou = asymmetric_3d_iou(sRT_1, sRT_2, size_1, size_2)

    return max_iou


def compute_IoU_matches(gt_class_ids, gt_sRT, gt_size, gt_handle_visibility,
                        pred_class_ids, pred_sRT, pred_size, pred_scores,
                        synset_names, iou_3d_thresholds, score_threshold=0):
    """ Find matches between NOCS prediction and ground truth instances.

    Args:
        size: 3D bounding box size
        bboxes: 2D bounding boxes

    Returns:
        gt_matches: 2-D array. For each GT box it has the index of the matched predicted box.
        pred_matches: 2-D array. For each predicted box, it has the index of the matched ground truth box.
        overlaps: IoU overlaps.
        indices:

    """
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    indices = np.zeros(0)
    if num_pred:
        # Sort predictions by score from high to low
        indices = np.argsort(pred_scores)[::-1]
        pred_class_ids = pred_class_ids[indices].copy()
        pred_size = pred_size[indices].copy()
        pred_sRT = pred_sRT[indices].copy()
    # compute IoU overlaps [pred_bboxs gt_bboxs]
    overlaps = np.zeros((num_pred, num_gt), dtype=np.float32)
    for i in range(num_pred):
        for j in range(num_gt):
            overlaps[i, j] = compute_3d_IoU(pred_sRT[i], gt_sRT[j], pred_size[i, :], gt_size[j],
                synset_names[pred_class_ids[i]], synset_names[gt_class_ids[j]], gt_handle_visibility[j])
    # loop through predictions and find matching ground truth boxes
    num_iou_3d_thres = len(iou_3d_thresholds)
    pred_matches = -1 * np.ones([num_iou_3d_thres, num_pred])
    gt_matches = -1 * np.ones([num_iou_3d_thres, num_gt])
    for s, iou_thres in enumerate(iou_3d_thresholds):
        for i in range(indices.shape[0]):
            # Find best matching ground truth box
            # 1. Sort matches by score
            sorted_ixs = np.argsort(overlaps[i])[::-1]
            # 2. Remove low scores
            low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
            if low_score_idx.size > 0:
                sorted_ixs = sorted_ixs[:low_score_idx[0]]
            # 3. Find the match
            for j in sorted_ixs:
                # If ground truth box is already matched, go to next one
                if gt_matches[s, j] > -1:
                    continue
                # If we reach IoU smaller than the threshold, end the loop
                iou = overlaps[i, j]
                if iou < iou_thres:
                    break
                # Do we have a match?
                if not pred_class_ids[i] == gt_class_ids[j]:
                    continue
                if iou > iou_thres:
                    gt_matches[s, j] = i
                    pred_matches[s, i] = j
                    break
    return gt_matches, pred_matches, overlaps, indices


def compute_RT_errors(sRT_1, sRT_2, class_id, handle_visibility, synset_names):
    """
    Args:
        sRT_1: [4, 4]. homogeneous affine transformation
        sRT_2: [4, 4]. homogeneous affine transformation

    Returns:
        theta: angle difference of R in degree
        shift: l2 difference of T in centimeter
    """
    # make sure the last row is [0, 0, 0, 1]
    if sRT_1 is None or sRT_2 is None:
        return -1
    try:
        assert np.array_equal(sRT_1[3, :], sRT_2[3, :])
        assert np.array_equal(sRT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(sRT_1[3, :], sRT_2[3, :])
        exit()

    R1 = sRT_1[:3, :3] / np.cbrt(np.linalg.det(sRT_1[:3, :3]))
    T1 = sRT_1[:3, 3]
    R2 = sRT_2[:3, :3] / np.cbrt(np.linalg.det(sRT_2[:3, :3]))
    T2 = sRT_2[:3, 3]
    # symmetric when rotating around y-axis
    if synset_names[class_id] in ['bottle', 'can', 'bowl'] or \
        (synset_names[class_id] == 'mug' and handle_visibility == 0):
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        cos_theta = y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2))
    else:
        R = R1 @ R2.transpose()
        cos_theta = (np.trace(R) - 1) / 2

    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
    shift = np.linalg.norm(T1 - T2) * 100
    result = np.array([theta, shift])

    return result


def compute_RT_overlaps(gt_class_ids, gt_sRT, gt_handle_visibility, pred_class_ids, pred_sRT, synset_names):
    """ Finds overlaps between prediction and ground truth instances.

    Returns:
        overlaps:num_pred,num_gt,2

    """
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    overlaps = np.zeros((num_pred, num_gt, 2))

    for i in range(num_pred):
        for j in range(num_gt):
            overlaps[i, j, :] = compute_RT_errors(pred_sRT[i], gt_sRT[j], gt_class_ids[j],
                                                  gt_handle_visibility[j], synset_names)
    return overlaps


def compute_RT_matches(overlaps, pred_class_ids, gt_class_ids, degree_thres_list, shift_thres_list):
    num_degree_thres = len(degree_thres_list)
    num_shift_thres = len(shift_thres_list)
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)

    pred_matches = -1 * np.ones((num_degree_thres, num_shift_thres, num_pred))
    gt_matches = -1 * np.ones((num_degree_thres, num_shift_thres, num_gt))

    if num_pred == 0 or num_gt == 0:
        return gt_matches, pred_matches

    assert num_pred == overlaps.shape[0]
    assert num_gt == overlaps.shape[1]
    assert overlaps.shape[2] == 2

    for d, degree_thres in enumerate(degree_thres_list):
        for s, shift_thres in enumerate(shift_thres_list):
            for i in range(num_pred):
                # Find best matching ground truth box
                # 1. Sort matches by scores from low to high
                sum_degree_shift = np.sum(overlaps[i, :, :], axis=-1)
                sorted_ixs = np.argsort(sum_degree_shift)
                # 2. Find the match
                for j in sorted_ixs:
                    # If ground truth box is already matched, go to next one
                    if gt_matches[d, s, j] > -1 or pred_class_ids[i] != gt_class_ids[j]:
                        continue
                    # If we reach IoU smaller than the threshold, end the loop
                    if overlaps[i, j, 0] > degree_thres or overlaps[i, j, 1] > shift_thres:
                        continue
                    gt_matches[d, s, j] = i
                    pred_matches[d, s, i] = j
                    break

    return gt_matches, pred_matches


def compute_ap_and_acc(pred_matches, pred_scores, gt_matches):
    # sort the scores from high to low
    assert pred_matches.shape[0] == pred_scores.shape[0]
    score_indices = np.argsort(pred_scores)[::-1]
    # pred_scores = pred_scores[score_indices]
    pred_matches = pred_matches[score_indices]
    precisions = np.cumsum(pred_matches > -1) / (np.arange(len(pred_matches)) + 1)
    recalls = np.cumsum(pred_matches > -1).astype(np.float32) / len(gt_matches)
    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])
    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])
    # compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    # accuracy
    acc = np.sum(pred_matches > -1) / len(pred_matches)

    return ap, acc


def compute_mAP(pred_results, out_dir, degree_thresholds=[180], shift_thresholds=[100],
                iou_3d_thresholds=[0.1], iou_pose_thres=0.1, use_matches_for_pose=False):
    """ Compute mean Average Precision.

    Returns:
        iou_aps:
        pose_aps:
        iou_acc:
        pose_acc:

    """
    synset_names = ['BG', 'bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
    num_classes = len(synset_names)
    degree_thres_list = list(degree_thresholds) + [360]
    num_degree_thres = len(degree_thres_list)
    shift_thres_list = list(shift_thresholds) + [100]
    num_shift_thres = len(shift_thres_list)
    iou_thres_list = list(iou_3d_thresholds)
    num_iou_thres = len(iou_thres_list)

    if use_matches_for_pose:
        assert iou_pose_thres in iou_thres_list

    # pre-allocate more than enough memory
    iou_aps = np.zeros((num_classes + 1, num_iou_thres))
    iou_acc = np.zeros((num_classes + 1, num_iou_thres))
    iou_pred_matches_all = [np.zeros((num_iou_thres, 30000)) for _ in range(num_classes)]
    iou_pred_scores_all = [np.zeros((num_iou_thres, 30000)) for _ in range(num_classes)]
    iou_gt_matches_all = [np.zeros((num_iou_thres, 30000)) for _ in range(num_classes)]
    iou_pred_count = [0 for _ in range(num_classes)]
    iou_gt_count = [0 for _ in range(num_classes)]

    pose_aps = np.zeros((num_classes + 1, num_degree_thres, num_shift_thres))
    pose_acc = np.zeros((num_classes + 1, num_degree_thres, num_shift_thres))
    pose_pred_matches_all = [np.zeros((num_degree_thres, num_shift_thres, 30000)) for _ in range(num_classes)]
    pose_pred_scores_all = [np.zeros((num_degree_thres, num_shift_thres, 30000)) for _ in range(num_classes)]
    pose_gt_matches_all = [np.zeros((num_degree_thres, num_shift_thres, 30000)) for _ in range(num_classes)]
    pose_pred_count = [0 for _ in range(num_classes)]
    pose_gt_count = [0 for _ in range(num_classes)]

    # loop over results to gather pred matches and gt matches for iou and pose metrics
    progress = 0
    for progress, result in enumerate(tqdm(pred_results)):
        gt_class_ids = result['gt_class_ids'].astype(np.int32)
        gt_sRT = np.array(result['gt_RTs'])
        gt_size = np.array(result['gt_scales'])
        gt_handle_visibility = result['gt_handle_visibility']

        pred_class_ids = result['pred_class_ids']
        pred_sRT = np.array(result['pred_RTs'])
        pred_size = result['pred_scales']
        pred_scores = result['pred_scores']

        if len(gt_class_ids) == 0 and len(pred_class_ids) == 0:
            continue

        for cls_id in range(1, num_classes):
            # get gt and predictions in this class
            cls_gt_class_ids = gt_class_ids[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros(0)
            cls_gt_sRT = gt_sRT[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros((0, 4, 4))
            cls_gt_size = gt_size[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros((0, 3))
            if synset_names[cls_id] != 'mug':
                cls_gt_handle_visibility = np.ones_like(cls_gt_class_ids)
            else:
                cls_gt_handle_visibility = gt_handle_visibility[gt_class_ids==cls_id] if len(gt_class_ids) else np.ones(0)

            cls_pred_class_ids = pred_class_ids[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros(0)
            cls_pred_sRT = pred_sRT[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 4, 4))
            cls_pred_size = pred_size[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 3))
            cls_pred_scores = pred_scores[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros(0)

            # calculate the overlap between each gt instance and pred instance
            iou_cls_gt_match, iou_cls_pred_match, _, iou_pred_indices = \
                compute_IoU_matches(cls_gt_class_ids, cls_gt_sRT, cls_gt_size, cls_gt_handle_visibility,
                                    cls_pred_class_ids, cls_pred_sRT, cls_pred_size, cls_pred_scores,
                                    synset_names, iou_thres_list)

            if len(iou_pred_indices):
                cls_pred_class_ids = cls_pred_class_ids[iou_pred_indices]
                cls_pred_sRT = cls_pred_sRT[iou_pred_indices]
                cls_pred_scores = cls_pred_scores[iou_pred_indices]

            num_pred = iou_cls_pred_match.shape[1]
            pred_start = iou_pred_count[cls_id]
            pred_end = pred_start + num_pred
            iou_pred_count[cls_id] = pred_end
            iou_pred_matches_all[cls_id][:, pred_start:pred_end] = iou_cls_pred_match
            cls_pred_scores_tile = np.tile(cls_pred_scores, (num_iou_thres, 1))
            assert cls_pred_scores_tile.shape[1] == num_pred
            iou_pred_scores_all[cls_id][:, pred_start:pred_end] = cls_pred_scores_tile
            num_gt = iou_cls_gt_match.shape[1]
            gt_start = iou_gt_count[cls_id]
            gt_end = gt_start + num_gt
            iou_gt_count[cls_id] = gt_end
            iou_gt_matches_all[cls_id][:, gt_start:gt_end] = iou_cls_gt_match


            if use_matches_for_pose:
                thres_ind = list(iou_thres_list).index(iou_pose_thres)
                iou_thres_pred_match = iou_cls_pred_match[thres_ind, :]
                cls_pred_class_ids = cls_pred_class_ids[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros(0)
                cls_pred_sRT = cls_pred_sRT[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros((0, 4, 4))
                cls_pred_scores = cls_pred_scores[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros(0)
                iou_thres_gt_match = iou_cls_gt_match[thres_ind, :]
                cls_gt_class_ids = cls_gt_class_ids[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros(0)
                cls_gt_sRT = cls_gt_sRT[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros((0, 4, 4))
                cls_gt_handle_visibility = cls_gt_handle_visibility[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros(0)

            RT_overlaps = compute_RT_overlaps(cls_gt_class_ids, cls_gt_sRT, cls_gt_handle_visibility,
                                              cls_pred_class_ids, cls_pred_sRT, synset_names)
            # if cls_id==3:
            #     print(RT_overlaps)
            pose_cls_gt_match, pose_cls_pred_match = compute_RT_matches(RT_overlaps, cls_pred_class_ids, cls_gt_class_ids,
                                                                        degree_thres_list, shift_thres_list)
            num_pred = pose_cls_pred_match.shape[2]
            pred_start = pose_pred_count[cls_id]
            pred_end = pred_start + num_pred
            pose_pred_count[cls_id] = pred_end
            pose_pred_matches_all[cls_id][:, :, pred_start:pred_end] = pose_cls_pred_match
            cls_pred_scores_tile = np.tile(cls_pred_scores, (num_degree_thres, num_shift_thres, 1))
            assert cls_pred_scores_tile.shape[2] == num_pred
            pose_pred_scores_all[cls_id][:, :, pred_start:pred_end] = cls_pred_scores_tile
            num_gt = pose_cls_gt_match.shape[2]
            gt_start = pose_gt_count[cls_id]
            gt_end = gt_start + num_gt
            pose_gt_count[cls_id] = gt_end
            pose_gt_matches_all[cls_id][:, :, gt_start:gt_end] = pose_cls_gt_match

    # trim zeros
    for cls_id in range(num_classes):
        # IoU
        iou_pred_matches_all[cls_id] = iou_pred_matches_all[cls_id][:, :iou_pred_count[cls_id]]
        iou_pred_scores_all[cls_id] = iou_pred_scores_all[cls_id][:, :iou_pred_count[cls_id]]
        iou_gt_matches_all[cls_id] = iou_gt_matches_all[cls_id][:, :iou_gt_count[cls_id]]
        # pose
        pose_pred_matches_all[cls_id] = pose_pred_matches_all[cls_id][:, :, :pose_pred_count[cls_id]]
        pose_pred_scores_all[cls_id] = pose_pred_scores_all[cls_id][:, :, :pose_pred_count[cls_id]]
        pose_gt_matches_all[cls_id] = pose_gt_matches_all[cls_id][:, :, :pose_gt_count[cls_id]]

    # compute 3D IoU mAP
    for cls_id in range(1, num_classes):
        for s, iou_thres in enumerate(iou_thres_list):
            iou_aps[cls_id, s], iou_acc[cls_id, s] = compute_ap_and_acc(iou_pred_matches_all[cls_id][s, :],
                                                                        iou_pred_scores_all[cls_id][s, :],
                                                                        iou_gt_matches_all[cls_id][s, :])
    iou_aps[-1, :] = np.mean(iou_aps[1:-1, :], axis=0)
    iou_acc[-1, :] = np.mean(iou_acc[1:-1, :], axis=0)
    # compute pose mAP
    for i, degree_thres in enumerate(degree_thres_list):
        for j, shift_thres in enumerate(shift_thres_list):
            for cls_id in range(1, num_classes):
                cls_pose_pred_matches_all = pose_pred_matches_all[cls_id][i, j, :]
                cls_pose_gt_matches_all = pose_gt_matches_all[cls_id][i, j, :]
                cls_pose_pred_scores_all = pose_pred_scores_all[cls_id][i, j, :]
                pose_aps[cls_id, i, j], pose_acc[cls_id, i, j] = compute_ap_and_acc(cls_pose_pred_matches_all,
                                                                                    cls_pose_pred_scores_all,
                                                                                    cls_pose_gt_matches_all)
            pose_aps[-1, i, j] = np.mean(pose_aps[1:-1, i, j])
            pose_acc[-1, i, j] = np.mean(pose_acc[1:-1, i, j])

    # save results to pkl
    result_dict = {}
    result_dict['iou_thres_list'] = iou_thres_list
    result_dict['degree_thres_list'] = degree_thres_list
    result_dict['shift_thres_list'] = shift_thres_list
    result_dict['iou_aps'] = iou_aps
    result_dict['pose_aps'] = pose_aps
    result_dict['iou_acc'] = iou_acc
    result_dict['pose_acc'] = pose_acc
    pkl_path = os.path.join(out_dir, 'mAP_Acc.pkl')
    with open(pkl_path, 'wb') as f:
        cPickle.dump(result_dict, f)
    return iou_aps, pose_aps, iou_acc, pose_acc


def plot_mAP(iou_aps, pose_aps, out_dir, iou_thres_list, degree_thres_list, shift_thres_list):
    """ Draw iou 3d AP vs. iou thresholds.
    """

    labels = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug', 'mean', 'nocs']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:pink', 'tab:olive', 'tab:purple', 'tab:red', 'tab:gray']
    styles = ['-', '-', '-', '-', '-', '-', '--', ':']

    fig, (ax_iou, ax_degree, ax_shift) = plt.subplots(1, 3, figsize=(8, 3.5))
    # IoU subplot
    ax_iou.set_title('3D IoU', fontsize=10)
    ax_iou.set_ylabel('Average Precision')
    ax_iou.set_ylim(0, 100)
    ax_iou.set_xlabel('Percent')
    ax_iou.set_xlim(0, 100)
    ax_iou.xaxis.set_ticks([0, 25, 50, 75, 100])
    ax_iou.grid()
    for i in range(1, iou_aps.shape[0]):
        ax_iou.plot(100*np.array(iou_thres_list), 100*iou_aps[i, :],
                    color=colors[i-1], linestyle=styles[i-1], label=labels[i-1])
    # rotation subplot
    ax_degree.set_title('Rotation', fontsize=10)
    ax_degree.set_ylim(0, 100)
    ax_degree.yaxis.set_ticklabels([])
    ax_degree.set_xlabel('Degree')
    ax_degree.set_xlim(0, 60)
    ax_degree.xaxis.set_ticks([0, 20, 40, 60])
    ax_degree.grid()
    for i in range(1, pose_aps.shape[0]):
        ax_degree.plot(np.array(degree_thres_list), 100*pose_aps[i, :len(degree_thres_list), -1],
                       color=colors[i-1], linestyle=styles[i-1], label=labels[i-1])
    # translation subplot
    ax_shift.set_title('Translation', fontsize=10)
    ax_shift.set_ylim(0, 100)
    ax_shift.yaxis.set_ticklabels([])
    ax_shift.set_xlabel('Centimeter')
    ax_shift.set_xlim(0, 10)
    ax_shift.xaxis.set_ticks([0, 5, 10])
    ax_shift.grid()
    for i in range(1, pose_aps.shape[0]):
        ax_shift.plot(np.array(shift_thres_list), 100*pose_aps[i, -1, :len(shift_thres_list)],
                      color=colors[i-1], linestyle=styles[i-1], label=labels[i-1])
    ax_shift.legend(loc='lower right', fontsize='small')
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(out_dir, 'mAP.jpg'))
    plt.close(fig)
    return


def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Args:
        coordinates_3d: [3, N]
        intrinsics: [3, 3]

    Returns:
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates


def align_rotation(sRT):
    """ Align rotations for symmetric objects.
    Args:
        sRT: 4 x 4
    """
    s = np.cbrt(np.linalg.det(sRT[:3, :3]))
    R = sRT[:3, :3] / s
    T = sRT[:3, 3]

    theta_x = R[0, 0] + R[2, 2]
    theta_y = R[0, 2] - R[2, 0]
    r_norm = math.sqrt(theta_x**2 + theta_y**2)
    s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                      [0.0,            1.0,  0.0           ],
                      [theta_y/r_norm, 0.0,  theta_x/r_norm]])
    rotation = R @ s_map
    aligned_sRT = np.identity(4, dtype=np.float32)
    aligned_sRT[:3, :3] = s * rotation
    aligned_sRT[:3, 3] = T
    return aligned_sRT


def draw_bboxes(img, img_pts, color):
    img_pts = np.int32(img_pts).reshape(-1, 2)
    # draw ground layer in darker color
    color_ground = (int(color[0]*0.7), int(color[1]*0.7), int(color[2]*0.7))
    for i, j in zip([4, 5, 6, 7], [5, 7, 4, 6]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_ground, 2)
    # draw pillars in minor darker color
    color_pillar = (int(color[0]*0.7), int(color[1]*0.7), int(color[2]*0.7))
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_pillar, 2)
    # draw top layer in original color
    color_top = (int(color[0]*0.9), int(color[1]*0.9), int(color[2]*0.9))
    for i, j in zip([0, 1, 2, 3], [1, 3, 0, 2]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_top, 2)

    for i, j in zip([ 1, 4,5], [ 5, 0, 4]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color, 2)

    return img

def draw_coordinate_system(img,sRT,size,cam_k,color=(255,0,0)):
    points=np.array([[0,0,0],[0.2,0,0],[0,0.2,0],[0,0,0.2]])
    nocs_scale=np.linalg.norm(size,ord=2)
    # points=points/nocs_scale
    points_3d=transform_coordinates_3d(points.T,sRT)
    points_2d=calculate_2d_projections(points_3d,cam_k)
    points_2d=np.int32(points_2d).reshape(-1,2)
    color=[(0,0,255),(0,255,0),(255,0,0)]
    for i in range(1,4):
        img=cv2.line(img,tuple(points_2d[0]),tuple(points_2d[i]),color[i-1],3)
    return img


def draw_detections(img, out_dir, data_name, img_id, intrinsics, pred_sRT, pred_size, pred_class_ids,
                    gt_sRT, gt_size, gt_class_ids, nocs_sRT, nocs_size, nocs_class_ids, draw_gt=True, draw_nocs=True):
    """ Visualize pose predictions.
    """
    out_path = os.path.join(out_dir, '{}_{}_pred.png'.format(data_name, img_id))

    # draw nocs results - BLUE color
    if draw_nocs:
        for i in range(nocs_sRT.shape[0]):
            if nocs_class_ids[i] in [1, 2, 4]:
                sRT = align_rotation(nocs_sRT[i, :, :])
            else:
                sRT = nocs_sRT[i, :, :]
            bbox_3d = get_3d_bbox(nocs_size[i, :], 0)
            transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
            projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
            img = draw_bboxes(img, projected_bbox, (255, 0, 0))
    # darw ground truth - GREEN color
    if draw_gt:
        for i in range(gt_sRT.shape[0]):
            if gt_class_ids[i] in [1, 2, 4]:
                sRT = align_rotation(gt_sRT[i, :, :])
            else:
                sRT = gt_sRT[i, :, :]
            bbox_3d = get_3d_bbox(gt_size[i, :], 0)
            transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
            projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
            img = draw_bboxes(img, projected_bbox, (0, 255, 0))
    # darw prediction - RED color
    for i in range(pred_sRT.shape[0]):
        if pred_class_ids[i] in [1, 2, 4]:
            sRT = align_rotation(pred_sRT[i, :, :])
        else:
            sRT = pred_sRT[i, :, :]
        bbox_3d = get_3d_bbox(pred_size[i, :], 0)
        transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
        projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
        img = draw_bboxes(img, projected_bbox, (0, 0, 255))

    cv2.imwrite(out_path, img)
    # cv2.imshow('vis', img)
    # cv2.waitKey(0)
