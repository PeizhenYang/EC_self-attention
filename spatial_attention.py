import torch
import torch.nn as nn
import math

class SpatialSelfAttention(nn.Module):
    def __init__(self, n_heads, in_channels, d_e):
        super(SpatialSelfAttention, self).__init__()
        self.n_heads = n_heads
        self.in_channels = in_channels
        self.d_e = d_e
        self.query_conv = nn.Linear(in_channels, n_heads*d_e, kernel_size=1)
        self.key_conv = nn.Linear(in_channels, n_heads*d_e, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, in_channels, seq_len = x.size()

        Q = self.query_conv(x).view(batch_size, self.n_heads, self.d_e, seq_len).permute(0, 1, 3, 2)
        K = self.query_conv(x).view(batch_size, self.n_heads, self.d_e, seq_len).permute(0, 1, 3, 2) #[batch_size, head, len, de]
        
        A = torch.matmul(Q, K.transpose(-2, -1))      # [batch_size, head, 1, 1]
        mask = torch.ones_like(A).triu(diagonal=1)
        A = A.masked_fill(mask==1, -float('inf')) 
        A = self.softmax(A / math.sqrt(self.d_e))
        
        x = x.view(batch_size, 1, seq_len*in_channels) # Reshape tensor
        A = A.view(batch_size, self.n_heads, seq_len, seq_len)
        H_att = torch.matmul(A, x)                   # [batchsize, head, seq_len, chan*len]
        #H_att = F.normalize(H_att, p=2, dim=-1)
        H_att = H_att.view(batch_size, self.n_heads*self.in_channels, seq_len) #reshape

        return H_att + x.squeeze()