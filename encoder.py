from feature_extractor import FeatureExtractor
import torch.nn as nn

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