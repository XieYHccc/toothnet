import torch

# 假设有 2 个 batch，每个 batch 有 16 颗牙齿，每颗牙齿是一个 3D 坐标
gt_centroids = torch.arange(1, 1 + 2 * 16 * 3).view(2, 16, 3).float()
print("gt_centroids shape:", gt_centroids.shape)
print("gt_centroids:\n", gt_centroids)

# 假设第 0, 1, 3, 5, 7, 10, 15 颗牙齿存在
gt_centroid_exists = torch.tensor([[1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
                                   [1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]], dtype=torch.bool)
print("\ngt_centroid_exists shape:", gt_centroid_exists.shape)
print("gt_centroid_exists:\n", gt_centroid_exists)

# 掩码筛选质心
selected = gt_centroids[gt_centroid_exists > 0, :]
print("\nSelected centroids shape:", selected.shape)
print("Selected centroids:\n", selected)