import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import models
import math

class SpatialSelfAttention(nn.Module):
    def __init__(self, n_heads, in_channels, d_e):
        super(SpatialSelfAttention, self).__init__()
        self.n_heads = n_heads
        self.in_channels = in_channels
        self.d_e = d_e
        self.query_conv = nn.Linear(in_channels, n_heads*d_e)
        self.key_conv = nn.Linear(in_channels, n_heads*d_e)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, seq_len, in_channels = x.size()
        residual = x
        
        Q_w, K_w = torch.mean(x, dim=1), torch.max(x, dim=1)[0]  # pooling [batchsize, chan]
        Q = self.query_conv(Q_w).view(batch_size, self.n_heads, self.d_e)
        K = self.key_conv(K_w).view(batch_size, self.n_heads, self.d_e)
        
        A = torch.matmul(Q, K.transpose(-2, -1))      # [batch_size, head, 1, 1]
        mask = torch.ones_like(A).triu(diagonal=1)
        A = A.masked_fill(mask==1, -float('inf')) 
        A = self.softmax(A / math.sqrt(self.d_e))
        
        x = x.view(batch_size, 1, seq_len, in_channels) # Reshape x tensor
        A = A.view(batch_size, self.n_heads, 1, 1)
        H_att = torch.matmul(A, x)                   # [batchsize, head, 1, chan]
        H_att = H_att.view(batch_size, self.n_heads*self.in_channels) # 2D reshape

        return H_att


class Predictor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Predictor, self).__init__()
        self.encoder = nn.Sequential(
            #FeatureExtractor(in_channels, out_channels),
            SpatialSelfAttention(n_heads=8, in_channels=out_channels, d_e=int(out_channels/8)),
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x):
        H_att = self.encoder(x)
        h_hat = self.decoder(H_att)
        return h_hat