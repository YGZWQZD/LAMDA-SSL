import torch.nn as nn
import torch.nn.functional as F
import torch
class Consistency(nn.Module):
    def __init__(self,reduction='mean',activation_1=None,activation_2=None):
        super().__init__()
        self.reduction = reduction
        self.activation_1=activation_1
        self.activation_2=activation_2
    def forward(self,logits_1,logits_2):
        if self.activation_1 is not None:
            logits_1=self.activation_1(logits_1)
        if self.activation_2 is not None:
            logits_2=self.activation_2(logits_2)
        F.mse_loss(logits_1, logits_2, reduction='mean')