import torch
import numpy as np
from sklearn.neighbors import KDTree
from toothnet.external.pointops2.functions import pointops

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

def get_tooth_centroids_np(pos_arr, labels):
    """
    Args:
        pos_arr (np.ndarray): A (N, 3) array of 3D coordinates.
        labels (np.ndarray): A (N,) array of integer (0, 16) labels corresponding to each point. 
    Returns:
        centroids (np.ndarray): A (16, 3) array of centroid's 3D coordinates
        exists (torch.Tensor): A (16,) array of tooth's existing mask 
    """

    # -1 means gingiva, 0-15 means 16 tooth
    labels = labels[:] - 1
    gt_cent_coords = []
    gt_cent_exists = []
    for class_idx in range(0, 16):
        cls_cond = labels==class_idx
        
        cls_verts = pos_arr[cls_cond]
        if cls_verts.shape[0]==0:
            gt_cent_coords.append(np.array([-10,-10,-10]))
            gt_cent_exists.append(0)
        else:
            centroid = np.mean(cls_verts, axis=0)
            gt_cent_coords.append(centroid)
            gt_cent_exists.append(1)
    
    gt_cent_coords = np.concatenate(gt_cent_coords, axis=0)
    gt_cent_exists = np.array(gt_cent_exists)

