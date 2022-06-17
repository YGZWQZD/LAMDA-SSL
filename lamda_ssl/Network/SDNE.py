import torch

class SDNE(torch.nn.Module):

    def __init__(self, input_dim, hidden_layers, device="cpu"):
        '''
        Structural Deep Network Embedding（SDNE）
        :param input_dim: 节点数量 node_size
        :param hidden_layers: AutoEncoder中间层数
        :param alpha: 对于1st_loss的系数
        :param beta: 对于2nd_loss中对非0项的惩罚
        :param device:
        '''
        super(SDNE, self).__init__()
        self.device = device
        input_dim_copy = input_dim
        layers = []
        for layer_dim in hidden_layers:
            layers.append(torch.nn.Linear(input_dim, layer_dim))
            layers.append(torch.nn.ReLU())
            input_dim = layer_dim
        self.encoder = torch.nn.Sequential(*layers)

        layers = []
        for layer_dim in reversed(hidden_layers[:-1]):
            layers.append(torch.nn.Linear(input_dim, layer_dim))
            layers.append(torch.nn.ReLU())
            input_dim = layer_dim
        # 最后加一层输入的维度
        layers.append(torch.nn.Linear(input_dim, input_dim_copy))
        layers.append(torch.nn.ReLU())
        self.decoder = torch.nn.Sequential(*layers)
        # torch中的只对weight进行正则真难搞啊
        # self.regularize = Regularization(self.encoder, weight_decay=gamma).to(self.device) + Regularization(self.decoder,weight_decay=gamma).to(self.device)


    def forward(self,X=None):
        '''
        输入节点的领接矩阵和拉普拉斯矩阵，主要计算方式参考论文
        :param A: adjacency_matrix, dim=(m, n)
        :param L: laplace_matrix, dim=(m, m)
        :return:
        '''
        Y = self.encoder(X)
        X_hat = self.decoder(Y)
        # loss_2nd 二阶相似度损失函数

        return Y,X_hat
