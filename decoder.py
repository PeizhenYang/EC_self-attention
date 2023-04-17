import torch.nn as nn

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