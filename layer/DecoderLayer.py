import torch
from torch import nn
from MultiHeadAttention import MultiHeadAttention

class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_size, dropout_prob=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(hidden_size, num_heads)  # 自注意力层
        self.dropout1 = nn.Dropout(dropout_prob)  # Dropout 层
        self.layer_norm1 = nn.LayerNorm(hidden_size)  # LayerNorm 层

        self.encoder_decoder_attention = MultiHeadAttention(hidden_size, num_heads)  # 编码器-解码器注意力层
        self.dropout2 = nn.Dropout(dropout_prob)  # Dropout 层
        self.layer_norm2 = nn.LayerNorm(hidden_size)  # LayerNorm 层

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, ff_size),  # 前馈层1
            nn.ReLU(),  # 激活函数
            nn.Linear(ff_size, hidden_size)  # 前馈层2
        )
        self.dropout3 = nn.Dropout(dropout_prob)  # Dropout 层
        self.layer_norm3 = nn.LayerNorm(hidden_size)  # LayerNorm 层
    
    def forward(self, x, encoder_output):
        # 自注意力子层
        self_attn_output = self.self_attention(x)  # (batch_size, seq_len, hidden_size)
        self_attn_output = self.dropout1(self_attn_output)  # Dropout
        out1 = self.layer_norm1(x + self_attn_output)  # 残差连接 + LayerNorm
        
        # 编码器-解码器注意力子层
        enc_dec_attn_output = self.encoder_decoder_attention(out1, encoder_output)  # (batch_size, seq_len, hidden_size)
        enc_dec_attn_output = self.dropout2(enc_dec_attn_output)  # Dropout
        out2 = self.layer_norm2(out1 + enc_dec_attn_output)  # 残差连接 + LayerNorm
        
        # 前馈神经网络子层
        ff_output = self.feed_forward(out2)  # (batch_size, seq_len, hidden_size)
        ff_output = self.dropout3(ff_output)  # Dropout
        out3 = self.layer_norm3(out2 + ff_output)  # 残差连接 + LayerNorm
        
        return out3

# 测试 DecoderLayer 的 main 函数
def main():
    batch_size = 2
    seq_len = 4
    hidden_size = 512
    num_heads = 8
    ff_size = 2048
    
    # 随机生成输入数据 (batch_size, seq_len, hidden_size)
    x = torch.randn(batch_size, seq_len, hidden_size)
    encoder_output = torch.randn(batch_size, seq_len, hidden_size)
    
    # 创建 DecoderLayer 模块
    decoder_layer = DecoderLayer(hidden_size, num_heads, ff_size)
    
    # 计算 DecoderLayer 输出
    output = decoder_layer(x, encoder_output)
    
    print("Input shape:", x.shape)
    print("Encoder output shape:", encoder_output.shape)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()