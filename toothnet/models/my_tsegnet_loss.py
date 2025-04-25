import torch
from toothnet.models.pointnet2_utils import square_distance

def masked_distance_loss(pred_distance, sample_xyz, centroid, centroid_exist_mask):
    """
    Args:
        pred_distance (torch.Tensor): [B, num_points], predicted distance from each sample point to the nearest GT centroid
        sample_xyz (torch.Tensor): [B, 3, num_points], 3D coordinates of sampled points
        centroid (torch.Tensor): [B, 3, max_centroids], padded GT centroid coordinates
        centroid_exist_mask (torch.Tensor): [B, max_centroids], binary mask indicating valid centroids (1 = valid, 0 = padded)

    Returns:
        loss (torch.Tensor): scalar smooth L1 loss between predicted and actual nearest centroid distances
    """
    B, _, N = sample_xyz.shape
    _, _, M = centroid.shape
    sample_xyz = sample_xyz.permute(0, 2, 1)
    centroid = centroid.permute(0, 2, 1)
    
    # Compute squared distances between each sample point and each centroid -> [B, N, M]
    dists = square_distance(sample_xyz, centroid)
    # Apply mask: assign large distance to invalid (padded) centroids
    invalid_mask = (1 - centroid_exist_mask).bool().unsqueeze(1)  # [B, 1, M]
    dists.masked_fill_(invalid_mask, float('inf'))
    # Get the square root of the minimum valid distance for each point -> [B, N]
    min_dists = torch.sqrt(dists.min(dim=-1).values)
    # Compute Smooth L1 loss between predicted and actual distances
    loss = torch.nn.functional.smooth_l1_loss(pred_distance, min_dists)
    return loss