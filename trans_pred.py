import torch
import torch.nn as nn
from decoder import DecoderLayer
from encoder import EncoderLayer
    

class Transformer(nn.Module):
    def __init__(self, input_dim,inputchannel, hidden_dim, num_layers):
        super(Transformer, self).__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(input_dim,inputchannel, hidden_dim) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(input_dim,hidden_dim) for _ in range(num_layers)])
        self.fc_layer = nn.Linear(input_dim, 1)

    def forward(self, x):
        encoder_out = x
        for layer in self.encoder_layers:
            print('1',encoder_out.shape)
            encoder_out = layer(encoder_out)
            print('1',encoder_out.shape)
        decoder_out = encoder_out
        for layer in self.decoder_layers:
            decoder_out = layer(decoder_out, encoder_out)
        pred_mean = torch.mean(decoder_out, dim=1)
        out = self.fc_layer(pred_mean)
        return out
