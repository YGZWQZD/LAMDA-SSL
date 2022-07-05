import torch.nn as nn
import torch.nn.functional as F
import torch
class KL_div(nn.Module):
    def __init__(self,softmax_1=True,softmax_2=True,reduction='mean'):
        super(KL_div, self).__init__()
        self.softmax_1=softmax_1
        self.softmax_2=softmax_2
        self.reduction=reduction

    def forward(self,logits_1,logits_2):#KL(p||q)
        if self.softmax_1:
            p = F.softmax(logits_1, dim=1)
            logp = F.log_softmax(logits_1, dim=1)
        else:
            p=logits_1
            logp=torch.log(p)

        if self.softmax_2:
            logq = F.log_softmax(logits_2, dim=1)
        else:
            logq=torch.log(logits_2)

        plogp = ( p *logp).sum(dim=1)
        plogq = ( p *logq).sum(dim=1)
        kl_div=plogp - plogq
        if self.reduction=='mean':
            kl_div=kl_div.mean()
        elif self.reduction=='sum':
            kl_div=kl_div.sum()
        return kl_div