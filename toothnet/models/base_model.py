import torch.nn.functional as F
from torch import nn

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__() 
        self.config = config

    def forward(self, inputs, targets=None):
        """
        Args:
            intputs(torch.Tensor): (B, 6, 16000) batched point features(position and normal)
            targets(torch.Tensor): (B, 1, 16000) batchedtarget labels
        Retruns:
            if targets is not None. return LossMap - a map from a named loss to a tensor storing the
            loss, otherwise, return standard predicted output
        """
        if targets is not None:
            return self.forward_training(inputs, targets)
        else:
            assert self.training == False
            outputs = self.forward_inference(inputs)
            if "cls_pred" in outputs:
                outputs["labels"] = self.postprocess(outputs["cls_pred"])
            return outputs
    
    def forward_training(self, inputs, targets):
        raise NotImplementedError

    def forward_inference(self, inputs):
        raise NotImplementedError

    def postprocess(self, cls_pred):
        probs = F.softmax(cls_pred, dim=1)
        pred_labels = probs.argmax(axis=1)
        pred_labels = pred_labels.cpu().detach().numpy()
        batch_size = pred_labels.shape[0]
        pred_labels.reshape(batch_size, 16000, 1)
        return pred_labels
