import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        print("feature",x.shape)
        return x


class SpatialSelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SpatialSelfAttention, self).__init__()
        self.query_layer = nn.Linear(input_dim, hidden_dim)
        self.key_layer = nn.Linear(input_dim, hidden_dim)
        self.softmax_layer = nn.Softmax(dim=-1)

    def forward(self, H):
        N, T = H.shape[0],H.shape[1]
        #print(H.shape)
        Q = self.query_layer(H)
        K = self.key_layer(H).transpose(-1,-2)
        #print(Q.size(),K.size())
        A = torch.matmul(Q, K)
        #A = torch.where(torch.eye(N, device=H.device), torch.zeros_like(A), A)
        A /= torch.sqrt(torch.tensor(K.size(-1)).float())
        #print(A.size())
        A = self.softmax_layer(A)
        #print(A.size(),H.size())
        H_ = torch.matmul(A, H)
        print("attention shape", H_.shape)
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
        print("weight",h_hat.shape)
        return h_hat


class EncoderLayer(nn.Module):
    def __init__(self, input_dim,input_channel, hidden_dim):
        super(EncoderLayer, self).__init__()
        self.attention_layer = SpatialSelfAttention(input_dim, hidden_dim)
        self.extractor_layer = FeatureExtractor(input_channel, input_channel)
        self.norm1 = nn.LayerNorm(input_dim)
        #self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        out = self.extractor_layer(x)
        attn_output = self.attention_layer(out)
        out = (out + attn_output)
        out = self.norm1(out)
        print("Encoder", out.shape)
        return out

class DecoderLayer(nn.Module):
    def __init__(self, input_dim,hidden_dim):
        super(DecoderLayer, self).__init__()
        self.self_attention_layer = SpatialSelfAttention(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        #self.attention_layer = SpatialSelfAttention(input_dim, input_dim)
        self.attention_layer = SpatialSelfAttention(input_dim, hidden_dim)


    def forward(self, x, encoder_out):
        #print(f'Decoder Layer Input: {x.shape}')
        out = self.norm1(x)
        attn_output = self.self_attention_layer(encoder_out)
        #print(f'Decoder Layer Self Attention Output: {attn_output.shape}')
        out = (out + attn_output)
        print("decoder", out.shape)
        # out = self.norm2(out)
        # attn_output, _ = self.attention_layer(out, encoder_out.transpose(1, 2), encoder_out.transpose(1, 2))
        # print(f'Decoder Layer Encoder Attention Output: {attn_output.shape}')
        # out = (out + attn_output)
        # out = self.norm3(out)
        return out


class Transformer(nn.Module):
    def __init__(self, input_dim, inputchannel, hidden_dim, num_layers):
        super(Transformer, self).__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(input_dim,inputchannel, hidden_dim) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(input_dim,hidden_dim) for _ in range(num_layers)])
        self.fc_layer = nn.Linear(input_dim, 1)

    def forward(self, x):
        encoder_out = x
        for layer in self.encoder_layers:
            print('11',encoder_out.shape)
            encoder_out = layer(encoder_out)
            print('12',encoder_out.shape)
        decoder_out = encoder_out
        print("decoder_out",decoder_out.shape)
        for layer in self.decoder_layers:
            decoder_out = layer(decoder_out, encoder_out)
        # pred_mean = torch.mean(decoder_out, dim=1, keepdim = True)
        # print("pred_mean",pred_mean.shape)
        out = self.fc_layer(decoder_out)
        print("tran",out.shape)
        return out


seq_len = 2000
batch_size = 6
input_dim = 20
hidden_dim = 65
num_layers = 2
num_heads = 4

x = torch.randn(batch_size,input_dim,seq_len)
model = Transformer(seq_len,input_dim,hidden_dim, num_layers)
output = model(x)
print(f'Output shape: {output.shape}')
print(output)

#train_x, test_x = x[]