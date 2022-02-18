from Semi_sklearn.Transform.Transformer import Transformer
import random
class Random_swap(Transformer):
    def __init__(self,n=1):
        super(Random_swap, self).__init__()
        self.n=n

    def swap(self,X):
        random_idx_1 = random.randint(0, len(X) - 1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(X) - 1)
            counter += 1
            if counter > 3:
                return X
        X[random_idx_1], X[random_idx_2] = X[random_idx_2], X[random_idx_1]
        return X

    def transform(self,X):
        for _ in range(self.n):
            X = self.swap(X)
        return X
