import torch.nn as nn
class Semi_supervised_loss(nn.Module):
    def __init__(self,lambda_u=1.0):
        super().__init__()
        self.lambda_u=lambda_u
    def foward(self,sup_loss,unsup_loss):
        return sup_loss+self.lambda_u*unsup_loss