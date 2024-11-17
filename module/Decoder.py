import torch
from torch import nn
from layer.DecoderLayer import DecoderLayer
from layer.Embedding import PositionalEmbedding, TokenEmbedding

class Decoder(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_size, num_layers, dropout_prob=0.1):
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_size, num_heads, ff_size, dropout_prob)
            for _ in range(num_layers)
        ])

    def forward(self, x, encoder_output):
        for layer in self.layers:
            x = layer(x, encoder_output)

        return x  # (batch_size, seq_len, hidden_size)
    
# 测试 Decoder 的 main 函数
def main():
    batch_size = 2
    seq_len = 4
    hidden_size = 512
    num_heads = 8
    ff_size = 2048
    num_layers = 6
    
    # 随机生成输入数据 (batch_size, seq_len, hidden_size)
    x = torch.randn(batch_size, seq_len, hidden_size)
    encoder_output = torch.randn(batch_size, seq_len, hidden_size)
    
    # 创建 Decoder 模块
    decoder = Decoder(hidden_size, num_heads, ff_size, num_layers)
    
    # 计算 Decoder 输出
    output = decoder(x, encoder_output)
    
    print("Input shape:", x.shape)
    print("Encoder output shape:", encoder_output.shape)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()