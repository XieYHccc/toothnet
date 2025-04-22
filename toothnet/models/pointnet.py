import torch
import torch.nn as nn
import torch.nn.functional as F

from toothnet.models.toothnet_loss import tooth_class_loss
from toothnet.loss_utils import LossMap
from toothnet.models.pointnet_utils import PointNetEncoder
from toothnet.models.base_model import BaseModel

def get_loss(gt_seg_label_1, sem_1):
    tooth_class_loss_1 = tooth_class_loss(sem_1, gt_seg_label_1, 17)
    return {
        "tooth_class_loss_1": (tooth_class_loss_1, 1),
    }

class PointNet(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.k = 17
        scale = 2
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=6, scale=scale)
        self.conv1 = torch.nn.Conv1d(1088 * scale, 512 * scale, 1)
        self.conv2 = torch.nn.Conv1d(512 * scale, 256 * scale, 1)
        self.conv3 = torch.nn.Conv1d(256 * scale, 128 * scale, 1)
        self.conv4 = torch.nn.Conv1d(128 * scale, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512 * scale)
        self.bn2 = nn.BatchNorm1d(256 * scale)
        self.bn3 = nn.BatchNorm1d(128 * scale)

    def forward_training(self, inputs, targets):
        logits = self._forward(inputs)
        loss_map = LossMap()
        loss_map.add_loss_by_dict(get_loss(targets, logits))
        return loss_map
    
    def forward_inference(self, inputs):
       logits = self._forward(inputs)
       outputs = {"cls_pred": logits}
       return outputs
    
    def _forward(self, inputs):
        x = inputs
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        # x = x.transpose(2, 1).contiguous()
        # x = F.log_softmax(x.view(-1, self.k), dim=-1)
        # x = x.view(batchsize, n_pts, self.k).permute(0, 2, 1)

        return x
    
if __name__ == '__main__':
    model = PointNet({})
    xyz = torch.rand(12, 3, 2048)
    (model(xyz))