import torch
from torch import nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, query, key, value, mask=None):
        # q\k\v's shape: (bs, seq_len, hidden_size)
        
        # 计算注意力分数
        # key.transpose(-1, -2) 将最后两个维度进行转置，以进行点积
        # attention_scores 形状: (batch_size, seq_len, seq_len)
        d_k = query.size(-1) # hidden_size, 其实表示q、k的维度
        attention_score = torch.matmul(query, key.transpose(-1, -2)) \
                         / torch.sqrt(torch.tensor(d_k))
        

        # 添加注意力掩码（seq_len, seq_len），掩码位置（1）的值为负无穷
        # 使用 masked_fill 将 mask == 0 的位置替换成 -float('inf')，即负无穷大
        # 后续计算 softmax 时，这些位置的值会变为 0，因此它们不会对输出产生影响。这样一来，模型的注意力机制就会“忽略”这些位置。
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -float('inf'))
        
        # 应用softmax函数，得到注意力权重
        # (bs, num_heads, seq_len, seq_len)
        attention_weights = torch.softmax(attention_score, dim=-1) 
        
        # 应用注意力权重对 value 进行加权求和
        # (bs, num_heads, seq_len, hidden_size)
        attention_output = torch.matmul(attention_weights, value)

        return attention_output

def test_attn():
    batch_size = 32
    seq_len = 512
    hidden_size = 1024
    
    query = torch.randn(batch_size, seq_len, hidden_size)  # (batch_size, seq_len, hidden_size)
    key = torch.randn(batch_size, seq_len, hidden_size)    # (batch_size, seq_len, hidden_size)
    value = torch.randn(batch_size, seq_len, hidden_size)  # (batch_size, seq_len, hidden_size)

    sdpa = ScaledDotProductAttention()
    output = sdpa(query, key, value)
    
    print("Query shape:", query.shape)
    print("Key shape:", key.shape)
    print("Value shape:", value.shape)
    print("Output shape:", output.shape)

if __name__ == "__main__":
	test_attn()