import torch.nn as nn
class Semi_Supervised_Loss(nn.Module):
    def __init__(self,lambda_u=1.0):
        # lambda_u: The weight of unsupervised loss.
        super().__init__()
        self.lambda_u=lambda_u
    def forward(self,sup_loss,unsup_loss):
        # sup_loss: The supervised loss.
        # unsup_loss: The unsupervised loss.
        return sup_loss+self.lambda_u*unsup_loss