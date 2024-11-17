import torch
from torch import nn
from MultiHeadAttention import MultiHeadAttention
from Embedding import PositionalEmbedding, TokenEmbedding

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_size, dropout_prob=0.1):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(hidden_size, num_heads)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.layer_norm1 = nn.LayerNorm(hidden_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.ReLU(),
            nn.Linear(ff_size, hidden_size)
        )
        self.dropout2 = nn.Dropout(dropout_prob)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        # 多头注意力子层
        attn_output = self.multi_head_attention(x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layer_norm1(x + attn_output)

        # 前馈神经网络子层
        ffn_output = self.feed_forward(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layer_norm2(out1 + ffn_output)

        return out2
    
def test_EncoderLayer():
    batch_size = 2
    seq_len = 4
    hidden_size = 512
    num_heads = 8
    ff_size = 2048

    x = torch.randn(batch_size, seq_len, hidden_size)

    mask = None
    encoderlayer =  EncoderLayer(hidden_size, num_heads, ff_size)
    out = encoderlayer(x, mask)

    print(x.shape)
    print(out.shape)  # (batch_size, seq_len, hidden_size)
    assert out.shape == (batch_size, seq_len, hidden_size)

if __name__ == "__main__":
    test_EncoderLayer()