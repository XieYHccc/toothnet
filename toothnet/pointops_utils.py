import torch
import numpy as np
from sklearn.neighbors import KDTree
from toothnet.external.pointops.functions import pointops

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def torch_to_numpy(cuda_arr):
    return cuda_arr.cpu().detach().numpy()

def farthest_point_sample_torch(points, npoint):
    if points.shape[0]<=npoint:
        raise "new fps error"
    points = torch.from_numpy(np.array(points)).type(torch.float).cuda()
    idx = pointops.furthestsampling(points, torch.tensor([points.shape[0]]).cuda().type(torch.int), torch.tensor([npoint]).cuda().type(torch.int))
    return torch_to_numpy(idx).reshape(-1)

def get_nearest_neighbor_idx(org_xyz, sampled_clusters, crop_num=4096):
    """
    Args:
        org_xyz => type np => B, N, 3
        sampled_clusters => type np => B, cluster_num, 3
    Retruns:
        return - B, cluster_num, 4096
    """
    cropped_all = []
    for batch_idx in range(org_xyz.shape[0]):
        cropped_points = []

        tree = KDTree(org_xyz[batch_idx,:,:], leaf_size=2)
        indexs = tree.query(sampled_clusters[batch_idx], k=crop_num, return_distance = False)
        cropped_all.append(indexs)
    return cropped_all

def get_indexed_features(features, cropped_indexes):
    """
    Args:
        features => type torch cuda/np => B, channel, N
        cropped indexes => type torch cuda/np => B, cluster_num, 4096
    Returns:
        cropped_item_ls => type torch cuda/np => new batch B, channel, 4096
    """
    cropped_item_ls = []
    for b_idx in range(len(cropped_indexes)):
        for cluster_idx in range(len(cropped_indexes[b_idx])):
            #cropped_point = torch.index_select(features[b_idx,:,:], 1, torch.tensor(cropped_indexes[b_idx][cluster_idx]).cuda())
            cropped_point = features[b_idx][:, cropped_indexes[b_idx][cluster_idx]]
            cropped_item_ls.append(cropped_point)
    if type(cropped_item_ls[0]) == torch.Tensor:
        cropped_item_ls = torch.stack(cropped_item_ls, dim=0)
    elif type(cropped_item_ls[0]) == np.ndarray:
        cropped_item_ls = np.stack(cropped_item_ls, axis=0)
    else:
        raise "someting unknwon type"
    return cropped_item_ls

def get_tooth_centroids(gt_coords, gt_seg_label):
    """
    Args:
        positions(torch.Tensor): (1, 3, 16000)
        labels(torch.Tensor): (1, 1, 16000)
    Returns:
        centroids(torch.Tensor): (1, 3, 16)
        exists(torch.Tensor): (1, 16)
    """
    device = gt_coords.device
    gt_coords = gt_coords.permute(0,2,1)
    gt_coords = gt_coords.view(-1,3)
    gt_seg_label = gt_seg_label.view(-1)

    gt_cent_coords = []
    gt_cent_exists = []
    for class_idx in range(0, 16):
        cls_cond = gt_seg_label==class_idx
        
        cls_sample_xyz = gt_coords[cls_cond, :]
        if cls_sample_xyz.shape[0]==0:
            gt_cent_coords.append(torch.tensor([-10, -10, -10], device=device, dtype=gt_coords.dtype))
            gt_cent_exists.append(torch.zeros(1, device=device, dtype=gt_coords.dtype))
        else:
            centroid = torch.mean(cls_sample_xyz, axis=0)
            gt_cent_coords.append(centroid)
            gt_cent_exists.append(torch.ones(1, device=device, dtype=gt_coords.dtype))

    gt_cent_coords = torch.stack(gt_cent_coords)
    gt_cent_coords = gt_cent_coords.view(1, *gt_cent_coords.shape)
    gt_cent_coords = gt_cent_coords.permute(0,2,1)
    gt_cent_exists = torch.stack(gt_cent_exists)
    gt_cent_exists = gt_cent_exists.view(1, -1)
    return gt_cent_coords, gt_cent_exists

def get_tooth_centroids1(positions, labels):
    """
    Args:
        positions(torch.Tensor): (B, 3, 16000)
        labels(torch.Tensor): (B, 1, 16000)
    Returns:
        centroids(torch.Tensor): (B, 3, 16)
        exists(torch.Tensor): (B, 16)
    """
    device = positions.device
    batch_size, _, _ = positions.shape
    positions = positions.permute(0, 2, 1)
    labels = labels.permute(0, 2, 1)

    cent_coords = []
    cent_exists = []
    for i in range(batch_size):
        mesh_positions = positions[i]
        mesh_labels = labels[i]
        for class_idx in range(0, 16):
            cls_mask = mesh_labels==class_idx
            cls_positions = mesh_positions[cls_mask, :]
            if cls_positions.shape[0]==0:
                cent_coords.append(torch.tensor([-10, -10, -10], device=device, dtype=positions.dtype))
                cent_exists.append(torch.zeros(1, device=device, dtype=positions.dtype))
            else:
                centroid = torch.mean(cls_positions, axis=0)
                cent_coords.append(centroid)
                cent_exists.append(torch.ones(1, device=device, dtype=positions.dtype))

    cent_coords = torch.stack(cent_coords)
    cent_coords = cent_coords.view(batch_size, 16, 3)
    cent_coords = cent_coords.permute(0, 2, 1)
    cent_exists = torch.stack(cent_exists)
    cent_exists = cent_exists.view(batch_size, 16)

    return cent_coords, cent_exists


def seg_label_to_cent(gt_coords, gt_seg_label):
    gt_coords = gt_coords.permute(0,2,1)
    gt_coords = gt_coords.view(-1,3)
    gt_seg_label = gt_seg_label.view(-1)

    gt_cent_coords = []
    gt_cent_exists = []
    for class_idx in range(0, 16):
        cls_cond = gt_seg_label==class_idx
        
        cls_sample_xyz = gt_coords[cls_cond, :]
        if cls_sample_xyz.shape[0]==0:
            gt_cent_coords.append(torch.from_numpy(np.array([-10,-10,-10])))
            gt_cent_exists.append(torch.zeros(1))
        else:
            centroid = torch.mean(cls_sample_xyz, axis=0)
            gt_cent_coords.append(centroid)
            gt_cent_exists.append(torch.ones(1))

    gt_cent_coords = torch.stack(gt_cent_coords)
    gt_cent_coords = gt_cent_coords.view(1, *gt_cent_coords.shape)
    gt_cent_coords = gt_cent_coords.permute(0,2,1)
    gt_cent_exists = torch.stack(gt_cent_exists)
    gt_cent_exists = gt_cent_exists.view(1, -1)

    return gt_cent_coords, gt_cent_exists

def seg_label_to_cent(gt_coords, gt_seg_label):
    gt_coords = gt_coords.permute(0,2,1)
    gt_coords = gt_coords.view(-1,3)
    gt_seg_label = gt_seg_label.view(-1)

    gt_cent_coords = []
    gt_cent_exists = []
    for class_idx in range(0, 16):
        cls_cond = gt_seg_label==class_idx
        
        cls_sample_xyz = gt_coords[cls_cond, :]
        if cls_sample_xyz.shape[0]==0:
            gt_cent_coords.append(torch.from_numpy(np.array([-10,-10,-10])))
            gt_cent_exists.append(torch.zeros(1))
        else:
            centroid = torch.mean(cls_sample_xyz, axis=0)
            gt_cent_coords.append(centroid)
            gt_cent_exists.append(torch.ones(1))

    gt_cent_coords = torch.stack(gt_cent_coords)
    gt_cent_coords = gt_cent_coords.view(1, *gt_cent_coords.shape)
    gt_cent_coords = gt_cent_coords.permute(0,2,1)
    gt_cent_exists = torch.stack(gt_cent_exists)
    gt_cent_exists = gt_cent_exists.view(1, -1)

    return gt_cent_coords, gt_cent_exists