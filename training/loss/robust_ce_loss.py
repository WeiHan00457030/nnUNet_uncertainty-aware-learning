import torch
from torch import nn, Tensor
import numpy as np


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def forward(self, input: Tensor, target: Tensor, uncertainty_map: torch.Tensor = None, weight:float = None, high_uncertainty:bool = None) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        
        if uncertainty_map is not None and uncertainty_map.numel() != 0:
            
            loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=self.ignore_index)
            # loss.shape: [batch_size,D,H,W] => (D,H,W) from (20,256,256) to (10,16,16) 
            loss = loss_fn(input, target.long())
            
            #ucnertainty,_ = uncertainty_map.max(dim=1)
            ucnertainty = uncertainty_map.sum(dim=1)
            
            if high_uncertainty:
                weighted_loss = loss * (1 + weight * (ucnertainty))
            else:
                weighted_loss = loss * (1 + weight * (1 - ucnertainty))
                
            return weighted_loss.mean()
        else:
            return super().forward(input, target.long())

    
    

class TopKLoss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()
