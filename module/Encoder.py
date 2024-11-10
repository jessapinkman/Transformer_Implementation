import torch
from torch import nn
from layer.EncoderLayer import EncoderLayer
from layer.Embedding import PositionalEmbedding, TokenEmbedding
import sys
import os


class Encoder(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_size, num_layers, dropout_prob=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # 堆叠多个encoderlayer
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_size, num_heads, ff_size, dropout_prob)
            for _ in range(num_layers)
        ])

        # token、positional embedding
        self.token_embedding = TokenEmbedding(20000, hidden_size)
        self.positional_encoding = PositionalEmbedding(20000, hidden_size)
        # self.positional_encoding = nn.Embedding(20000, hidden_size)


    def forward(self, x, mask=None):
        # x's shape: (bs, seq_len, hidden_size)
        
        # 1. 引入位置编码
        x = self.token_embedding(x) + self.positional_encoding(torch.arange(x.size(1), device=x.device))

        # 2. 堆叠 encoderlayer
        for layer in self.layers:
            x = layer(x, mask)
        
        return x  # (bs, seq_len, hidden_size)

# 测试 Encoder 的 main 函数
def main():
    batch_size = 2
    seq_len = 4
    hidden_size = 512
    num_heads = 8
    ff_size = 2048
    num_layers = 6
    
    # 随机生成输入数据 (batch_size, seq_len, hidden_size)
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # 创建 Encoder 模块
    encoder = Encoder(hidden_size, num_heads, ff_size, num_layers)
    
    # 计算 Encoder 输出
    output = encoder(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()

