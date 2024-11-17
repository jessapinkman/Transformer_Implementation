import torch 
from torch import nn

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size # d_model in paper
        self.num_heads = num_heads  # 8 in paper
        self.head_dim = hidden_size // num_heads # 每个头的维度，二者必须整除 dk in paper
        print(self.head_dim, self.num_heads, self.hidden_size)
        # 初始化qkv的投影矩阵，将输入词向量线性变换为Q、K、V，要求维度保持一致
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        # 输出线性层，将拼接后的多头注意力输出变换为所需的输出维度，要求维度保持一致
        self.output_proj = nn.Linear(hidden_size, hidden_size)

        # 初始化位置编码
        self.positional_encoding = nn.Embedding(20000, hidden_size)

    def forward(self, hidden_state, mask=None):
        # hidden_state's shape: (bs, seq_len, hidden_size)
        
        # 1. 线性变换，将 hidden_state线性变换为 Q, K, V
        # (bs, seq_len, hidden_size)
        q = self.q_proj(hidden_state)
        k = self.k_proj(hidden_state)
        v = self.v_proj(hidden_state)

        # 2. 切分Q, K, V
        # (bs, num_heads, seq_len, head_dim)
        q = q.reshape(hidden_state.size(0), -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(hidden_state.size(0), -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(hidden_state.size(0), -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # q = self.spilt_head(q)

        # 3. 计算注意力分数
        # (bs, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim).to(hidden_state.device))

        # softmax
        attention_probs = torch.softmax(attn_scores, dim=-1) #(bs, num_heads, seq_len, seq_len)

        # 4. 计算注意力值 v (bs, num_heads, seq_len, head_dim) 相当于两个值的后两个维度相乘
        output = torch.matmul(attention_probs, v) #(bs, num_heads, seq_len, head_dim)

        # 5. concatenate 对多头注意力输出进行拼接
        output = output.transpose(1, 2).reshape(hidden_state.size(0), -1, \
                        self.head_dim * self. num_heads)
        
        # 6. 线性变换，将 output 线性变换为 (bs, seq_len, hidden_size)
        output = self.output_proj(output)
        return output
    
    def spilt_head(self, x):
        bs = x.size(0)
        # x (bs, seq_len, hidden_size)
        # 将hidden_size分割魏num_head、head_dim
        return x.reshape(bs, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # 返回形状: (batch_size, num_heads, seq_len, head_dim)
    
def test_MHA():
    bs = 128
    seq_len = 512
    hidden_size = 512
    num_heads = 8

    hidden_state = torch.randn(bs, seq_len, hidden_size)
    mha = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
    output = mha(hidden_state)
    print("Input shape:", hidden_state.shape)
    print("Output shape:", output.shape)
    
if __name__ == "__main__":
	test_MHA()





