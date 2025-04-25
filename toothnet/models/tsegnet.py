import torch
import numpy as np
from sklearn.cluster import DBSCAN
from toothnet.models.tsegnet_centroid import TSegNetCentroid
from toothnet.models.tsegnet_seg import TSegNetSegmentation
from toothnet.models.pointnet2_utils import square_distance
from toothnet.models.base_model import BaseModel
from toothnet.models.tsegnet_loss import centroid_loss, segmentation_loss
from toothnet.utils.pointops_utils import torch_to_numpy, get_nearest_neighbor_idx, get_indexed_features
from toothnet.utils.loss_utils import LossMap

def get_tooth_centroids_batch(positions, labels):
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
                cent_coords.append(torch.from_numpy(np.array([-10,-10,-10])))
                cent_exists.append(torch.zeros(1))
            else:
                centroid = torch.mean(cls_positions, axis=0)
                cent_coords.append(centroid)
                cent_exists.append(torch.ones(1))

    cent_coords = torch.stack(cent_coords)
    cent_coords = cent_coords.view(batch_size, 16, 3)
    cent_coords = cent_coords.permute(0, 2, 1)
    cent_exists = torch.stack(cent_exists)
    cent_exists = cent_exists.view(batch_size, 16)

    return cent_coords, cent_exists

def get_tooth_centroids(gt_coords, gt_seg_label):
    """
    Args:
        positions(torch.Tensor): (1, 3, 16000)
        labels(torch.Tensor): (1, 1, 16000)
    Returns:
        centroids(torch.Tensor): (1, 3, 16)
        exists(torch.Tensor): (1, 16)
    """
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

        gt_centroids, gt_centroid_exists = get_tooth_centroids(inputs[:, :3, :].cpu(), gt_labels)
        gt_centroids_label = torch.arange(0, 16).view(1, -1).cuda() + 1 # (1, 16)
        gt_centroid_exists = gt_centroid_exists.view(1, -1) # B, 16
        gt_centroids = gt_centroids.permute(0, 2, 1) # (B, 16, 3)
        gt_centroids = gt_centroids[gt_centroid_exists>0, :]
        gt_centroids_label = gt_centroids_label[gt_centroid_exists>0]
        gt_centroids = gt_centroids.unsqueeze(dim=0)
        gt_centroids_label = gt_centroids_label.unsqueeze(dim=0)
        gt_centroids = gt_centroids.permute(0,2,1)
        gt_centroids = gt_centroids.cuda() # B, 3, 16
        gt_centroids_label = gt_centroids_label.cuda() # B, 16

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
        inputs => [B, 6, 16000] : point features
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
        dbscan_results = DBSCAN(eps=0.03, min_samples=2).fit(moved_points)

        center_points = []
        for label in np.unique(dbscan_results.labels_):
            if label == -1: continue
            center_points.append(moved_points[dbscan_results.labels_==label].mean(axis=0))
        center_points = np.array(center_points)
        if center_points.shape[0] == 0:
            print(f"[WARNING] DBSCAN found no clusters. Using fallback center. moved points num: {moved_points.shape[0]}")
            center = moved_points.mean(axis=0, keepdims=True)
            center_points = center[None, :, :]  # shape (1, 1, 3)
        else:
            center_points = center_points[None, :, :]  # shape (1, N, 3)

        rand_indexes = np.random.permutation(center_points.shape[1])[:8]
        center_points = center_points.transpose(0,2,1)[:,:,rand_indexes].transpose(0,2,1)
        
        nn_crop_indexes = get_nearest_neighbor_idx(torch_to_numpy(l0_xyz.permute(0,2,1)), center_points, 3072)

        cropped_input_ls = get_indexed_features(inputs, nn_crop_indexes)
        cropped_feature_ls = get_indexed_features(l0_points, nn_crop_indexes)
        ddf = self._get_ddf(cropped_input_ls[:,:3,:].permute(0,2,1), center_points)
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

