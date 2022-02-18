import torch.nn as nn
import torch.nn.functional as F
import torch


class TextRCNN(nn.Module):

    def __init__(self, n_vocab,embedding_dim=300, padding_idx=None, hidden_size=256, num_layers=1, pad_size=32,
                 dropout=0.0, pretrained_embeddings=None,num_class=2):
        super(TextRCNN, self).__init__()

        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(n_vocab, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.maxpool = nn.MaxPool1d(pad_size)
        self.fc = nn.Linear(hidden_size * 2 + embedding_dim, num_class)


    def forward(self, x):
        print(x.shape)
        print(x.dtype)
        embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)

        return out