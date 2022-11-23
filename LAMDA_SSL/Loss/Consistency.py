import torch.nn as nn
import torch.nn.functional as F
import torch

class Consistency(nn.Module):
    def __init__(self,reduction='mean'):
        # >> Parameter
        # >> - reduction: How to handle the output.
        super().__init__()
        self.reduction = reduction

    def forward(self,logits_1,logits_2):
        # >> forward(logits_1,logits_2): Perform loss calculations.
        # >> - logits_1: The first input to compute consistency.
        # >> - logits_2: The second input to compute consistency.
        assert logits_1.size() == logits_2.size()
        return F.mse_loss(torch.softmax(logits_1, dim=-1), torch.softmax(logits_2, dim=-1), reduction=self.reduction)