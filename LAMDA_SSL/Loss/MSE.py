import torch.nn as nn
import torch.nn.functional as F

class MSE(nn.Module):
    def __init__(self,reduction='mean',activation_1=None,activation_2=None):
        super().__init__()
        # >> Parameter
        # >> - reduction: How to handle the output.
        # >> - activation_1: The activation function to process on the first input.
        # >> - activation_2: The activation function to process on the second input.
        self.reduction = reduction
        self.activation_1=activation_1
        self.activation_2=activation_2
    def forward(self,logits_1,logits_2):
        # >> forward(logits_1,logits_2): Perform loss calculations.
        # >> - logits_1: The first input to compute consistency.
        # >> - logits_2: The second input to compute consistency.
        if self.activation_1 is not None:
            logits_1=self.activation_1(logits_1)
        if self.activation_2 is not None:
            logits_2=self.activation_2(logits_2)
        return F.mse_loss(logits_1, logits_2, reduction=self.reduction)