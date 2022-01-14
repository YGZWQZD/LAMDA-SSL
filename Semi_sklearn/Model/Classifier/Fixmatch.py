from Semi_sklearn.Base.InductiveEstimator import InductiveEstimator
from Semi_sklearn.Base.SemiDeepModelMixin import SemiDeepModelMixin
from Semi_sklearn.Data_loader.SemiTrainDataloader import SemiTrainDataLoader
from Semi_sklearn.Data_loader.SemiTestDataloader import SemiTestDataLoader
from Semi_sklearn.Opitimizer.SemiOptimizer import SemiOptimizer
from Semi_sklearn.Scheduler.SemiScheduler import SemiLambdaLR
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
                 network=None,
                 epoch=1,
                 num_it_epoch=None,
                 num_it_total=None,
                 optimizer=None,
                 scheduler=None,
                 device='cpu',
                 threshold=None,
                 lambda_u=None,
                 mu=None,
                 ema=None,
                 T=None,
                 weight_decay=None
                 ):
        SemiDeepModelMixin.__init__(self,train_dataset=train_dataset,
                                    test_dataset=test_dataset,
                                    train_dataloader=train_dataloader,
                                    test_dataloader=test_dataloader,
                                    augmentation=augmentation,
                                    network=network,
                                    epoch=epoch,
                                    num_it_epoch=num_it_epoch,
                                    num_it_total=num_it_total,
                                    mu=mu,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    device=device
                                    )
        self.ema=ema
        self.lambda_u=lambda_u
        self.threshold=threshold
        self.T=T
        self.weight_decay=weight_decay
        if self.ema is not None:
            self.ema.init_ema(model=network,device=self.device)
        if isinstance(self.augmentation,dict):
            self.weakly_augmentation=self.augmentation['weakly_augmentation']
            self.strong_augmentation = self.augmentation['strongly_augmentation']
        elif isinstance(self.augmentation,list):
            self.weakly_augmentation = self.augmentation[0]
            self.strong_augmentation = self.augmentation[1]
        elif isinstance(self.augmentation,tuple):
            self.weakly_augmentation,self.strong_augmentation=self.augmentation
        else:
            self.weakly_augmentation = self.augmentation
            self.strong_augmentation = self.augmentation
        self.normalization=self.augmentation['normalization']


        if isinstance(self.optimizer,SemiOptimizer):
            no_decay = ['bias', 'bn']
            grouped_parameters = [
                {'params': [p for n, p in self.network.named_parameters() if not any(
                    nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
                {'params': [p for n, p in self.network.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            self.optimizer=self.optimizer.init_optimizer(params=grouped_parameters)

        if isinstance(self.scheduler,SemiLambdaLR):
            self.scheduler=self.scheduler.init_scheduler(optimizer=self.optimizer)

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

    def predict(self,test_X=None,test_dataset=None):
        return SemiDeepModelMixin.predict(self,test_X=test_X,test_dataset=test_dataset)

