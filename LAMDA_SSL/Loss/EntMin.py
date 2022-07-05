import torch.nn as nn
import torch.nn.functional as F
import torch
class EntMin(nn.Module):
    def __init__(self, reduction='mean', activation=None):
        # >> - reduction: How to handle the output.
        # >> - activation: The activation function to process on the logits.
        super(EntMin, self).__init__()
        self.activation=activation
        self.reduction=reduction

    def forward(self,logits):
        # >> -logits: The logits to calculate the loss.
        if self.activation is not None:
            p=self.activation(logits)
        else:
            p=logits
        log_pred = F.log_softmax(logits, dim=-1)
        loss = torch.sum(-p * log_pred, dim=-1)
        if self.reduction=='mean':
            loss=loss.mean()
        elif self.reduction=='sum':
            loss=loss.sum()
        return loss