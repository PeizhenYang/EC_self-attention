import torch
import torch.nn as nn
import math

class SpatialSelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SpatialSelfAttention, self).__init__()
        self.query_layer = nn.Linear(input_dim, hidden_dim)
        self.key_layer = nn.Linear(input_dim, hidden_dim)
        self.softmax_layer = nn.Softmax(dim=-1)

    def forward(self, H):
        N, T = H.shape[0],H.shape[1]
        print(H.shape)
        Q = self.query_layer(H)
        K = self.key_layer(H).transpose(-1,-2)
        print(Q.size(),K.size())
        A = torch.matmul(Q, K)
        #A = torch.where(torch.eye(N, device=H.device), torch.zeros_like(A), A)
        A /= torch.sqrt(torch.tensor(K.size(-1)).float())
        print(A.size())
        A = self.softmax_layer(A)
        print(A.size(),H.size())
        H_ = torch.matmul(A, H)
        return H_ + H