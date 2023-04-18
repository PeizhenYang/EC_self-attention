import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=1, padding=1):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


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


class TemporalWeightedSum(nn.Module):
    def __init__(self, n_channels):
        super(TemporalWeightedSum, self).__init__()
        self.fc1 = nn.Linear(n_channels, n_channels)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_channels, 1)

    def forward(self, x):
        a = self.fc2(self.relu(self.fc1(x)))           # [batch_size, seq_len, 1]
        a = F.softmax(a, dim=1)
        h_hat = torch.sum(a*x, dim=1) # [batch_size, n_channels]
        return h_hat


class EncoderLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(EncoderLayer, self).__init__()
        self.attention_layer = nn.MultiheadAttention(hidden_dim, num_heads)
        self.extractor_layer = FeatureExtractor(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        out = self.extractor_layer(x).transpose(0, 1)
        attn_output, _ = self.attention_layer(out, out, out)
        out = out + attn_output.transpose(0, 1)
        out = self.norm1(out)
        out2 = self.norm2(out + self.extractor_layer(out).transpose(0, 1))
        return out2

class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(DecoderLayer, self).__init__()
        self.self_attention_layer = nn.MultiheadAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention_layer = nn.MultiheadAttention(hidden_dim, num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x, encoder_out):
        out = self.norm1(x)
        attn_output, _ = self.self_attention_layer(out, out, out)
        out = out + attn_output
        out = self.norm2(out)
        attn_output, _ = self.attention_layer(out, encoder_out, encoder_out)
        out = out + attn_output
        out = self.norm3(out)
        return out
    

class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads):
        super(Transformer, self).__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(input_dim, hidden_dim, num_heads) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(hidden_dim, num_heads) for _ in range(num_layers)])
        #self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_layer = nn.Linear(input_dim, 1)

    def forward(self, x):
        encoder_out = x.transpose(0, 1)
        for layer in self.encoder_layers:
            encoder_out = layer(encoder_out)
        decoder_out = encoder_out
        for layer in self.decoder_layers:
            decoder_out = layer(decoder_out, encoder_out)
        pred_mean = torch.mean(decoder_out, dim=1)
        out = self.fc_layer(pred_mean)
        return out
    
x = torch.rand(16,20,128)
tf = Transformer(128, 64, 2, 4)
op=tf(x)
print(op.size())