from Semi_sklearn.Base.InductiveEstimator import InductiveEstimator
from Semi_sklearn.Base.SemiDeepModelMixin import SemiDeepModelMixin
from Semi_sklearn.Data_loader.SemiTrainDataloader import SemiTrainDataLoader
from Semi_sklearn.Data_loader.SemiTestDataloader import SemiTestDataLoader
import torch.nn.functional as F
import torch
def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])
def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])
class Fixmatch(InductiveEstimator,SemiDeepModelMixin):
    def __init__(self,train_dataset=None,test_dataset=None,
                 train_dataloader=None,
                 test_dataloader=None,
                 augmentation=None,
                 weakly_augmentation=None,
                 strong_augmentation=None,
                 normalization=None,
                 network=None,
                 epoch=1,
                 optimizer=None,
                 scheduler=None,
                 device=None,
                 threshold=None,
                 lambda_u=None,
                 mu=None,
                 ema=None,
                 T=None
                 ):
        SemiDeepModelMixin.__init__(self,train_dataset=train_dataset,
                                    test_dataset=test_dataset,
                                    train_dataloader=train_dataloader,
                                    test_dataloader=test_dataloader,
                                    augmentation=augmentation,
                                    network=network,
                                    epoch=epoch,
                                    mu=mu,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    device=device
                                    )
        self.ema=ema
        self.lambda_u=lambda_u
        self.threshold=threshold
        self.normalization=normalization
        self.T=T
        if self.ema is not None:
            self.ema.init_ema(model=network,device=self.device)
        if weakly_augmentation is not None:
            self.weakly_augmentation=weakly_augmentation
            self.strong_augmentation=strong_augmentation
        elif isinstance(self.augmentation,dict):
            self.weakly_augmentation=self.augmentation['weakly_augmentation']
            self.strong_augmentation = self.augmentation['strong_augmentation']
        elif isinstance(self.augmentation,list):
            self.weakly_augmentation = self.augmentation[0]
            self.strong_augmentation = self.augmentation[1]
        elif isinstance(self.augmentation,tuple):
            self.weakly_augmentation,self.strong_augmentation=self.augmentation
        else:
            self.weakly_augmentation = self.augmentation
            self.strong_augmentation = self.augmentation
        self.normalization=self.augmentation['normalization'] if self.augmentation['normalization'] is not None else self.normalization

    def train(self,lb_X,lb_y,ulb_X,*args,**kwargs):
        w_lb_X=self.weakly_augmentation.fit_transform(lb_X)
        w_ulb_X=self.weakly_augmentation.fit_transform(ulb_X)
        s_ulb_X=self.strong_augmentation.fit_transform(ulb_X)
        batch_size = w_lb_X.shape[0]
        inputs=torch.cat((w_lb_X, w_ulb_X, s_ulb_X))
        inputs = interleave(inputs, 2 * self.mu + 1)
        logits = self.network(inputs)
        logits = de_interleave(logits, 2 * self.mu + 1)
        logits_x = logits[:batch_size]
        logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
        result=(logits_x,lb_y,logits_u_w,logits_u_s)
        return result



    def get_loss(self,train_result,*args,**kwargs):
        logits_x, lb_y, logits_u_w, logits_u_s = train_result
        Lx = F.cross_entropy(logits_x, lb_y, reduction='mean')

        pseudo_label = torch.softmax(logits_u_w.detach() / self.T, dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.threshold).float()

        Lu = (F.cross_entropy(logits_u_s, targets_u,
                              reduction='none') * mask).mean()

        loss = Lx + self.lambda_u * Lu
        return loss

    def backward(self,loss,*args,**kwargs):
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        if self.ema:
            self.ema.update(self.network)
        self.network.zero_grad()

    def estimate(self,X,*args,**kwargs):
        X=self.normalization.fit_transform(X)
        if self.ema is not None:
            outputs=self.ema.ema(X)
        else:
            outputs = self.network(X)
        return outputs


    def get_predict_result(self,y_est,*args,**kwargs):
        max_probs,y_pred=torch.max(y_est, dim=-1)
        return y_pred



