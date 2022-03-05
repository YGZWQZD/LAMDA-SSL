from sklearn.metrics.pairwise import rbf_kernel
class RBF_kernel:
    def __init__(self,gamma):
        self.gamma=gamma
    def __call__(self,X,Y):
        return rbf_kernel(X,Y,gamma=self.gamma)
