import torch.nn as nn
from spatial_attention import SpatialSelfAttention

class DecoderLayer(nn.Module):
    def __init__(self, input_dim,hidden_dim):
        super(DecoderLayer, self).__init__()
        self.self_attention_layer = SpatialSelfAttention(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.attention_layer = SpatialSelfAttention(input_dim, input_dim)


    def forward(self, x, encoder_out):
        print(f'Decoder Layer Input: {x.shape}')
        out = self.norm1(x)
        attn_output = self.self_attention_layer(encoder_out)
        print(f'Decoder Layer Self Attention Output: {attn_output.shape}')
        out = (out + attn_output)
        # out = self.norm2(out)
        # attn_output, _ = self.attention_layer(out, encoder_out.transpose(1, 2), encoder_out.transpose(1, 2))
        # print(f'Decoder Layer Encoder Attention Output: {attn_output.shape}')
        # out = (out + attn_output)
        # out = self.norm3(out)
        return out