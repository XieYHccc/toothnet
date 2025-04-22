import torch.nn.functional as F
from torch import nn

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__() 
        self.config = config

    def forward(self, inputs, targets=None):
        if targets is not None:
            return self.forward_training(inputs, targets)
        else:
            assert self.training == False
            outputs = self.forward_inference(inputs)
            if "cls_pred" in outputs:
                outputs["labels"] = self.postprocess(outputs["cls_pred"])
            return outputs
    
    def forward_training(self, inputs, targets):
        """
        Args:
            intputs: [B, 6, 16000] batched point features(position and normal)
            targets: target labels
        Retruns:
            LossMap: mapping from a named loss to a tensor storing the
            loss
        """
        raise NotImplementedError

    def forward_inference(self, inputs):
        """
        Args:
            intputs: (B, 6, 16000) batched point features(position and normal)
        Retruns:
            outputs: a dict. all output attributes should be shape (B, 16000, x)
        """
        raise NotImplementedError

    def postprocess(self, cls_pred):
        probs = F.softmax(cls_pred, dim=1)
        pred_labels = probs.argmax(axis=1)
        pred_labels = pred_labels.cpu().detach().numpy()
        batch_size = pred_labels.shape[0]
        pred_labels.reshape(batch_size, 16000, 1)
        return pred_labels
