import torch.nn as nn
from feature_extractor import FeatureExtractor
from spatial_attention import SpatialSelfAttention

class EncoderLayer(nn.Module):
    def __init__(self, input_dim,input_channel, hidden_dim):
        super(EncoderLayer, self).__init__()
        self.attention_layer = SpatialSelfAttention(input_dim, hidden_dim)
        self.extractor_layer = FeatureExtractor(input_channel, input_channel)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        out = self.extractor_layer(x)
        attn_output = self.attention_layer(out)
        out = (out + attn_output)
        out = self.norm1(out)
        #out2 = self.norm2(out + self.extractor_layer(out.permute(1,3,0,2)).transpose(1, 2))
        return out