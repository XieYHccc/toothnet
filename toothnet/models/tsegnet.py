import torch
import numpy as np
from sklearn.cluster import DBSCAN
from toothnet.models.tsegnet_centroid import TSegNetCentroid
from toothnet.models.tsegnet_seg import TSegNetSegmentation
from toothnet.models.pointnet2_utils import square_distance
from toothnet.models.base_model import BaseModel
from toothnet.models.tsegnet_loss import centroid_loss, segmentation_loss
from toothnet.pointops_utils import torch_to_numpy, get_nearest_neighbor_idx, get_indexed_features, get_tooth_centroids, seg_label_to_cent
from toothnet.loss_utils import LossMap

class TSegNet(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.cent_module = TSegNetCentroid()
        self.seg_module = TSegNetSegmentation()

        if self.config["run_tooth_segmentation_module"]:
            self.run_seg_module = True
        else:
            self.run_seg_module = False

    def forward_training(self, inputs, targets):
        B, C, N = inputs.shape
        gt_labels = targets

        gt_centroids, gt_centroid_exists = get_tooth_centroids(inputs[:, :3, :], gt_labels)
        gt_centroids_label = torch.arange(0, 16).view(1, -1).cuda() + 1 # (1, 16)
        gt_centroid_exists = gt_centroid_exists.view(1, -1)
        gt_centroids = gt_centroids.permute(0, 2, 1) # (B, 16, 3)
        gt_centroids = gt_centroids[gt_centroid_exists>0, :]
        gt_centroids_label = gt_centroids_label[gt_centroid_exists>0]
        gt_centroids = gt_centroids.unsqueeze(dim=0)
        gt_centroids_label = gt_centroids_label.unsqueeze(dim=0)
        gt_centroids = gt_centroids.permute(0,2,1)
        gt_centroids = gt_centroids.cuda() # B, 3, 14
        gt_centroids_label = gt_centroids_label.cuda() # B, 14

        outputs = self._forward(inputs)
        if self.run_seg_module:
            cluster_gt_seg_label = get_indexed_features(targets, outputs["nn_crop_indexes"])
            outputs["cluster_gt_seg_label"] = cluster_gt_seg_label  
        
        loss_map = LossMap()
        loss_map.add_loss_by_dict(self._get_loss(outputs, {
                "seg_label": targets,
                "centroid_coords": gt_centroids,
                "centroid_labels": gt_centroids_label,
            } ))

        return loss_map
    
    def forward_inference(self, inputs):
        return super().forward_inference(inputs)
    
    def _forward(self, inputs):
        """
        inputs => [B, 6, 24000] : point features
        """
        B, C, N = inputs.shape
        outputs = {}

        l0_points, l3_points, l0_xyz, l3_xyz, offset_result, dist_result = self.cent_module(inputs)
        outputs.update({
            "l0_points": l0_points, 
            "l3_points":l3_points, 
            "l0_xyz": l0_xyz, 
            "l3_xyz": l3_xyz, 
            "offset_result":offset_result, 
            "dist_result":dist_result
        })
        if not self.run_seg_module: return outputs

        moved_points = torch_to_numpy(l3_xyz + offset_result).T.reshape(-1,3)
        moved_points = moved_points[torch_to_numpy(dist_result).reshape(-1)<0.3,:]
        dbscan_results = DBSCAN(eps=0.05, min_samples=3).fit(moved_points, 3)

        center_points = []
        for label in np.unique(dbscan_results.labels_):
            if label == -1: continue
            center_points.append(moved_points[dbscan_results.labels_==label].mean(axis=0))
        center_points = np.array(center_points)
        center_points = center_points[None, :, :]

        rand_indexes = np.random.permutation(center_points.shape[1])[:8]
        center_points = center_points.transpose(0,2,1)[:,:,rand_indexes].transpose(0,2,1)
        
        nn_crop_indexes = get_nearest_neighbor_idx(torch_to_numpy(l0_xyz.permute(0,2,1)), center_points, 3072)

        cropped_input_ls = get_indexed_features(inputs, nn_crop_indexes)
        cropped_feature_ls = get_indexed_features(l0_points, nn_crop_indexes)
        ddf = self.get_ddf(cropped_input_ls[:,:3,:].permute(0,2,1), center_points)
        cropped_feature_ls = torch.cat([cropped_input_ls[:,:3,:], cropped_feature_ls, ddf], axis=1)
        pd_1, weight_1, pd_2, id_pred = self.seg_module(cropped_feature_ls)
        outputs.update({
            "pd_1":pd_1, "weight_1":weight_1, "pd_2":pd_2, "id_pred":id_pred,
            "center_points":center_points, "cropped_feature_ls":cropped_feature_ls,
            "nn_crop_indexes":nn_crop_indexes
        })

        return outputs

    
    def _get_ddf(self, cropped_coord, center_points):
        B, N, C  = cropped_coord.shape
        
        center_points = torch.from_numpy(center_points).cuda()
        ddf = square_distance(cropped_coord, center_points.permute(1,0,2))
        ddf = torch.sqrt(ddf)
        ddf *= (-4)
        ddf = torch.exp(ddf)
        ddf = ddf.permute(0,2,1)
        return ddf
    
    def _get_loss(self, outputs, gt):
        losses = {}
        dist_loss, cent_loss, chamf_loss = centroid_loss(
            outputs["offset_result"], outputs["l3_xyz"], outputs["dist_result"], gt["centroid_coords"]
        )
        losses.update({
            "dist_loss": (dist_loss, 1),
            "cent_loss": (cent_loss, 1),
            "chamf_loss": (chamf_loss, 0.1),
        })

        if self.config["run_tooth_segmentation_module"] is False: return losses

        sqd = square_distance(torch.from_numpy(outputs["center_points"]).cuda(),gt["centroid_coords"].permute(0,2,1))  # 1, N, 3 X 1, M, 3 => 1, N, M
        sqd_argmin =  sqd.argmin(axis=2).reshape(-1)
        pred_centerpoint_gt_label_ls = gt["centroid_labels"][:, sqd_argmin] # 1, N, M => 1, N
        
        cluster_gt_seg_bin_label_ls = torch.zeros_like(outputs["cluster_gt_seg_label"]).cuda()
        for i in range(outputs["cluster_gt_seg_label"].shape[0]):
            cluster_gt_seg_bin_label_ls[i, 0, pred_centerpoint_gt_label_ls[0][i]==outputs["cluster_gt_seg_label"][i][0]+1] = 1

        seg_1_loss, seg_2_loss, id_pred_loss = segmentation_loss(outputs["pd_1"], outputs["weight_1"], outputs["pd_2"], outputs["id_pred"],
        pred_centerpoint_gt_label_ls, cluster_gt_seg_bin_label_ls)
        losses.update({
            "seg_1_loss":(seg_1_loss,1), 
            "seg_2_loss":(seg_2_loss,1), 
            "id_pred_loss":(id_pred_loss,1)
        })
        
        return losses

