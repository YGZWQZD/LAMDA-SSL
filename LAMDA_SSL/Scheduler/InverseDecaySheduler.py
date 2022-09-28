from LAMDA_SSL.Base.LambdaLR import LambdaLR

class InverseDecaySheduler(LambdaLR):
    def __init__(self, initial_lr, gamma=10, power=0.75, max_iter=1000):
        self.initial_lr=initial_lr
        self.gamma=gamma
        self.power=power
        self.max_iter=max_iter

    def _lr_lambda(self, current_step):
        return self.initial_lr * ((1 + self.gamma * min(1.0, current_step / float(self.max_iter))) ** (- self.power))