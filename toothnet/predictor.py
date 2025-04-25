import torch
import numpy as np
from sklearn.neighbors import KDTree
from toothnet.models.base_model import BaseModel
from toothnet.utils.io_utils import load_mesh, get_jaw_type_from_path
from toothnet.utils.pointops_utils import pc_normalize, farthest_point_sample_torch

class Predictor:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def predict(self, vert_arr, jaw_type, output_names):
        """
        Args:
            vert_array (np.ndarray): (N, 6), position + normal
            jaw_type (str): upper or lower
            output_names (list): expected outputs
        
        Returns:
            outputs (dict): with key in output names
        """
        # normalize and downsample
        vert_arr[:, :3] = pc_normalize(vert_arr[:, :3])
        samples_indexes = farthest_point_sample_torch(vert_arr[:, :3], 16000)
        original_feat = vert_arr.copy().astype("float32")
        sampled_feat = original_feat[samples_indexes]

        sampled_feat_cuda = torch.from_numpy(sampled_feat)
        sampled_feat_cuda = sampled_feat_cuda.permute(1, 0)
        sampled_feat_cuda = sampled_feat_cuda.unsqueeze(0)
        sampled_feat_cuda = sampled_feat_cuda.cuda()
        with torch.no_grad():
            outputs = self.model(sampled_feat_cuda)

        results = {}
        for key in output_names:
            results[key] = outputs[key]

        # interpolation to original point cloud
        tree = KDTree(sampled_feat[:, :3], leaf_size=2)
        near_points = tree.query(original_feat[:,:3], k=1, return_distance=False)

        if "labels" in output_names and "labels" in outputs:
            labels = outputs["labels"].squeeze()[near_points.reshape(-1)]
            # map label from 0-16 to fdi system
            labels[labels>=9] += 2
            labels[labels>0] += 10
            if jaw_type == "lower":
                labels[labels>0] += 20
            elif jaw_type == "upper":
                pass
            else:
                raise "Jaw type error"
            results["labels"] = labels
        
        return results
    
    def predict_from_path(self, filepath, output_names):
       vert_arr = load_mesh(filepath)
       jaw_type = get_jaw_type_from_path(filepath)
       return self.predict(vert_arr, jaw_type, output_names)



        