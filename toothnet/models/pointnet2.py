import torch.nn as nn
import torch.nn.functional as F
from toothnet.models.pointnet2_utils import PointNetSetAbstractionMsg, PointNetFeaturePropagation
from toothnet.models.toothnet_loss import tooth_class_loss
from toothnet.models.base_model import BaseModel
from toothnet.utils.loss_utils import LossMap

def get_loss(gt_seg_label_1, sem_1):
    tooth_class_loss_1 = tooth_class_loss(sem_1, gt_seg_label_1, 17)
    return {
        "tooth_class_loss_1": (tooth_class_loss_1, 1),
    }

class PointNet2(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        input_feature_num = 6
        scale = 4
        # target point 개수, ball query radius, maximun sample in ball 개수, input feature 개수(position + 각각의 feature vector), MLP 개수, group_all False
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.025, 0.05], [32, 64], input_feature_num,
                                             [[32 * scale, 32 * scale], [32 * scale, 32 * scale]])
        self.sa2 = PointNetSetAbstractionMsg(512, [0.05, 0.1], [32, 64], 32 * scale + 32 * scale,
                                             [[64 * scale, 128 * scale], [64 * scale, 128 * scale]])
        self.sa3 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [32, 64], 128 * scale + 128 * scale,
                                             [[196 * scale, 256 * scale], [196 * scale, 256 * scale]])

        self.fp3 = PointNetFeaturePropagation((512 + 256) * scale, [256 * scale, 256 * scale])
        self.fp2 = PointNetFeaturePropagation((256 + 64) * scale, [128 * scale, 128 * scale])
        self.fp1 = PointNetFeaturePropagation((128 * scale) + input_feature_num, [64 * scale, 32 * scale])

        self.offset_conv_1 = nn.Conv1d(32 * scale, 16, 1)
        self.offset_bn_1 = nn.BatchNorm1d(16)
        self.dist_conv_1 = nn.Conv1d(32 * scale, 16, 1)
        self.dist_bn_1 = nn.BatchNorm1d(16)

        self.offset_conv_2 = nn.Conv1d(16, 3, 1)
        self.dist_conv_2 = nn.Conv1d(16, 1, 1)

        self.cls_conv_1 = nn.Conv1d(32 * scale, 17, 1)
        self.cls_bn_1 = nn.BatchNorm1d(17)
        self.cls_conv_2 = nn.Conv1d(17, 17, 1)

        nn.init.zeros_(self.offset_conv_2.weight)
        nn.init.zeros_(self.dist_conv_2.weight)

        # prediction part
        self.conv1 = nn.Conv1d(32, 16, 1)
        self.bn1 = nn.BatchNorm1d(16)
    
    def forward_traning(self, inputs, targets):
        logits = self._forward(inputs)
        loss_map = LossMap()
        loss_map.add_loss_by_dict(get_loss(targets, logits))
        return loss_map
    
    def forward_inference(self, inputs):
        logits = self._forward(inputs)
        outputs = {"cls_pred": logits}
        return outputs
    
    def _forward(self, inputs):
        xyz = inputs
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        # x = F.relu(self.bn1(self.conv1(l0_points)))

        offset_result = F.relu(self.offset_bn_1(self.offset_conv_1(l0_points)))
        offset_result = self.offset_conv_2(offset_result)

        dist_result = F.relu(self.dist_bn_1(self.dist_conv_1(l0_points)))
        dist_result = self.dist_conv_2(dist_result)

        # output = [l0_points, l3_points, l0_xyz, l3_xyz, offset_result, dist_result]
        cls_pred = F.relu(self.cls_bn_1(self.cls_conv_1(l0_points)))
        cls_pred = self.cls_conv_2(cls_pred)

        return cls_pred
    
class get_model(nn.Module):
    def __init__(self, config):
        super(get_model, self).__init__()
        self.cls_pred = True
        input_feature_num=6
        scale = 4
        # target point 개수, ball query radius, maximun sample in ball 개수, input feature 개수(position + 각각의 feature vector), MLP 개수, group_all False 
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.025, 0.05], [32, 64], input_feature_num, [[32*scale, 32*scale], [32*scale, 32*scale]])
        self.sa2 = PointNetSetAbstractionMsg(512, [0.05, 0.1], [32, 64], 32*scale+32*scale, [[64*scale, 128*scale], [64*scale, 128*scale]])
        self.sa3 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [32, 64], 128*scale+128*scale, [[196*scale, 256*scale], [196*scale, 256*scale]])
        

        self.fp3 = PointNetFeaturePropagation((512+256)*scale, [256*scale, 256*scale])
        self.fp2 = PointNetFeaturePropagation((256+64)*scale, [128*scale, 128*scale])
        self.fp1 = PointNetFeaturePropagation((128*scale)+input_feature_num, [64*scale, 32*scale])

        self.offset_conv_1 = nn.Conv1d(32*scale,16, 1)
        self.offset_bn_1 = nn.BatchNorm1d(16)
        self.dist_conv_1 = nn.Conv1d(32*scale,16, 1)
        self.dist_bn_1 = nn.BatchNorm1d(16)
        
        self.offset_conv_2 = nn.Conv1d(16,3, 1)
        self.dist_conv_2 = nn.Conv1d(16,1, 1)

        if self.cls_pred:
            self.cls_conv_1 = nn.Conv1d(32*scale,17, 1)
            self.cls_bn_1 = nn.BatchNorm1d(17)
            self.cls_conv_2 = nn.Conv1d(17,17, 1)

        nn.init.zeros_(self.offset_conv_2.weight)
        nn.init.zeros_(self.dist_conv_2.weight)

        #prediction part
        self.conv1 = nn.Conv1d(32, 16, 1)
        self.bn1 = nn.BatchNorm1d(16)
        
    #input으로, batch, channel(xyz + 기타등등), sample in batch 이렇게 와야 한다.
    def forward(self, xyz_in):
        xyz = xyz_in
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        #x = F.relu(self.bn1(self.conv1(l0_points)))
        
        offset_result = F.relu(self.offset_bn_1(self.offset_conv_1(l0_points)))
        offset_result = self.offset_conv_2(offset_result)

        dist_result = F.relu(self.dist_bn_1(self.dist_conv_1(l0_points)))
        dist_result = self.dist_conv_2(dist_result)

        output = [l0_points, l3_points, l0_xyz, l3_xyz, offset_result, dist_result]
        
        if self.cls_pred:
            cls_pred = F.relu(self.cls_bn_1(self.cls_conv_1(l0_points)))
            cls_pred = self.cls_conv_2(cls_pred)
            output.append(cls_pred)

        return output