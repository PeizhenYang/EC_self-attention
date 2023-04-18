import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=1, padding=1):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        print(x.shape)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        print(x.shape)
        return x


class SpatialSelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SpatialSelfAttention, self).__init__()
        self.query_layer = nn.Linear(input_dim, hidden_dim)
        self.key_layer = nn.Linear(input_dim, hidden_dim)
        self.softmax_layer = nn.Softmax(dim=-1)

    def forward(self, H):
        N, T = H.shape[0],H.shape[1]
        print(H.shape)
        Q = self.query_layer(H.transpose(-1,-2))
        K = self.key_layer(H.transpose(-1,-2))
        print(Q.size(),K.size())
        A = torch.matmul(Q.transpose(-1, -2), K)
        #A = torch.where(torch.eye(N, device=H.device,dtype=torch.bool), torch.zeros_like(A), A)
        A /= torch.sqrt(torch.tensor(K.size(-1)).float())
        print(A.size())
        A = self.softmax_layer(A)
        print(A.size(),H.size())
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
        a = torch.softmax(a, dim=1)
        h_hat = torch.sum(a*x, dim=1) # [batch_size, n_channels]
        return h_hat


class EncoderLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(EncoderLayer, self).__init__()
        self.attention_layer = SpatialSelfAttention(hidden_dim, hidden_dim)
        self.extractor_layer = FeatureExtractor(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        print(f'Encoder Layer Input: {x.shape}')
        out = self.extractor_layer(x).transpose(1, 2)
        print(f'Encoder Layer Extractor Output: {out.shape}')
        attn_output = self.attention_layer(out)
        print(f'Encoder Layer Attention Output: {attn_output.shape}')
        print(out.shape,attn_output.shape)
        out = (out + attn_output).permute(1, 0,3,2)
        out = self.norm1(out)
        out2 = self.norm2(out + self.extractor_layer(out.permute(1,3,0,2)).transpose(1, 2))
        return out2


class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(DecoderLayer, self).__init__()
        self.self_attention_layer = SpatialSelfAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention_layer = SpatialSelfAttention(hidden_dim, num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        def forward(self, x, encoder_out):
            print(f'Decoder Layer Input: {x.shape}')
            out = self.norm1(x.permute(1, 2, 0))
            attn_output, _ = self.self_attention_layer(out, out, out)
            print(f'Decoder Layer Self Attention Output: {attn_output.shape}')
            out = (out + attn_output).permute(2, 0, 1)
            out = self.norm2(out)
            attn_output, _ = self.attention_layer(out, encoder_out.transpose(1, 2), encoder_out.transpose(1, 2))
            print(f'Decoder Layer Encoder Attention Output: {attn_output.shape}')
            out = (out + attn_output).permute(1, 2, 0)
            out = self.norm3(out)
            return out


class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads):
        super(Transformer, self).__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(input_dim, hidden_dim, num_heads) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(hidden_dim, num_heads) for _ in range(num_layers)])
        self.fc_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        encoder_out = x.transpose(0,1)
        for layer in self.encoder_layers:
            print('1',encoder_out.shape)
            encoder_out = layer(encoder_out.transpose(1,2))
            print('1',encoder_out.shape)
        decoder_out = encoder_out
        for layer in self.decoder_layers:
            decoder_out = layer(decoder_out, encoder_out)
        pred_mean = torch.mean(decoder_out, dim=1)
        out = self.fc_layer(pred_mean)
        return out


seq_len = 10
batch_size = 1
input_dim = 64
hidden_dim = 64
num_layers = 2
num_heads = 4

x = torch.randn(input_dim,1,seq_len,batch_size)


in_channels = 10
out_channels = 64


#print(output_fe)


# ws = SpatialSelfAttention(input_dim, hidden_dim)
# output_ws = ws(x)
# print(output_sa)


model = Transformer(input_dim, hidden_dim, num_layers, num_heads)
output = model(x)
print(f'Output shape: {output.shape}')
