import torch
import numpy as np
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