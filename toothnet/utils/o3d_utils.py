import open3d as o3d
import numpy as np

def np_to_pcd(pos_arr, color):
    """
    Convert a numpy array of 3D points into an Open3D point cloud with a uniform color.

    Args:
        pos_arr (np.ndarray): A (N, 3) array of 3D coordinates.
        color (list or np.ndarray): A single RGB color [r, g, b], each in range 0–1.

    Returns:
        o3d.geometry.PointCloud: Open3D point cloud with the specified color.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos_arr)
    pcd.colors = o3d.utility.Vector3dVector([color]*len(pcd.points))
    return pcd

def np_to_pcd_with_label(arr, label_arr):
    """
    Convert a labeled point array into a color-coded Open3D point cloud.
    
    Args:
        arr (np.ndarray): A (N, 3) array of 3D coordinates.
        label_arr (np.ndarray): A (N,) array of integer labels corresponding to each point.

    Returns:
        o3d.geometry.PointCloud: Open3D point cloud with color-coded labels.
    """
    arr = np.concatenate([arr[:,:3], label_arr.reshape(-1,1)],axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:,:3])
    
    palte = np.array([
        [255,153,153], #gingiva
        [153,76,0], [153,153,0], [76,153,0], [0,153,153], [0,0,153], [153,0,153], [153,0,76], [64,64,64],
        [255,128,0], [153,153,0], [76,153,0], [0,153,153], [0,0,153], [153,0,153], [153,0,76],[64,64,64]
    ])/255

    palte[9:] *= 0.4
    arr = arr.copy()
    arr[:, 3] %= palte.shape[0]
    label_colors = np.zeros((arr.shape[0], 3))
    for idx, palte_color in enumerate(palte):
        label_colors[arr[:, 3]==idx] = palte_color
    pcd.colors = o3d.utility.Vector3dVector(label_colors)
    return pcd

