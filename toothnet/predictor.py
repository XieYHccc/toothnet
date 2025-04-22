import torch
import numpy as np
from sklearn.neighbors import KDTree
from toothnet.models.base_model import BaseModel
from toothnet.io_utils import load_mesh, get_jaw_type_from_path
from toothnet.pointops_utils import pc_normalize, farthest_point_sample_torch

class Predictor:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def predict(self, vert_arr, jaw_type, output_names):
        """
        Args:
            vert_array: np.ndarray(n, 6), position + normal
            jaw_type: str, upper or lower
            output_names: list, expected outputs
        
        Returns:
            outputs: dict with key in output names
        """
        # normalize and downsample
        pos_arr = vert_arr[:, :3]
        pos_arr = pc_normalize(pos_arr)
        samples_indexes = farthest_point_sample_torch(pos_arr, 16000)
        original_feat = vert_arr
        sampled_feat = original_feat[samples_indexes]

        sampled_feat = torch.from_numpy(sampled_feat)
        sampled_feat = sampled_feat.permute(1, 0)
        inputs = {"feat":sampled_feat}
        with torch.no_grad():
            outputs = self.model(inputs)

        # interpolation to original point cloud
        tree = KDTree(sampled_feat[:,:3], leaf_size=2)
        near_points = tree.query(original_feat[:,:3], k=1, return_distance=False)
        results = {}
        for key in output_names:
            results[key] = output_names[key][near_points.reshape(-1)]

        if "labels" in output_names and "labels" in outputs:
            labels = results["labels"]
            # map label from 0-16 to fdi system
            labels[labels>=9] += 2
            labels[labels>0] += 10
            if jaw_type == "lower":
                labels[labels>0] += 20
            elif jaw_type == "upper":
                pass
            else:
                raise "Jaw type error"
        
        return results
    
    def predict(self, filepath, output_names):
       vert_arr = load_mesh(filepath)
       jaw_type = get_jaw_type_from_path(filepath)
       return self.predict(vert_arr, jaw_type, output_names)



        