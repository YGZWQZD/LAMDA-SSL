import torch.nn as nn
import torch.nn.functional as F
import torch
class KL_div(nn.Module):
    def __init__(self,with_logit_1=True,with_logit_2=True):
        super(KL_div, self).__init__()
        self.with_logit_1=with_logit_1
        self.with_logit_2=with_logit_2
    def foward(self,logits_1,logits_2):#KL(p||q)
        if self.with_logit_1:
            p = F.softmax(logits_1, dim=1)
            logp = F.log_softmax(logits_1, dim=1)
        else:
            p=logits_1
            logp=torch.log(p)
        if self.with_logit_2:
            logq = F.log_softmax(logits_2, dim=1)
        else:
            logq=torch.log(logits_2)

        qlogp = ( p *logp).sum(dim=1).mean(dim=0)
        qlogq = ( p *logq).sum(dim=1).mean(dim=0)
        return qlogp - qlogq