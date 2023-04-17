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
        N, T = H.size()
        Q = self.query_layer(H)
        K = self.key_layer(H)
        
        A = torch.matmul(Q, K.transpose(-1, -2))
        A = torch.where(torch.eye(N, device=H.device), torch.zeros_like(A), A)
        A /= torch.sqrt(torch.tensor(K.size(-1)).float())
        A = self.softmax_layer(A)
        H_ = torch.matmul(A, H)
        
        return H_ + H