import torch
from torch import nn
from torch.nn.modules.utils import to_2tuple
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, x):
        embedded = self.embedding(x)
        return embedded

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, hidden_size):
        super(PositionalEmbedding, self).__init__()
        self.hidden_size = hidden_size

        # 创建位置编码表，大小为 (max_len, hidden_size)
        # position: (max_len, 1)，表示序列中的位置索引，例如 [[0.], [1.], [2.], ...]
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # div_term: (hidden_size / 2)，用于计算位置编码的分母
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        
        # 初始化位置编码矩阵 pe 为零矩阵，大小为 (max_len, hidden_size)
        pe = torch.zeros(max_len, hidden_size)
        
        # 计算位置编码矩阵，广播机制将 dive_term 扩展为 (1, hidden_size )
        # 偶数索引列使用 sin 函数
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数索引列使用 cos 函数
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 将位置编码矩阵注册为 buffer，模型训练时不会更新它
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 的形状: (batch_size, seq_len, hidden_size)
        seq_len = x.size(1)
        
        # 将位置编码加到输入张量上
        # self.pe[:seq_len, :] 的形状为 (seq_len, hidden_size)
        # unsqueeze(0) 使其形状变为 (1, seq_len, hidden_size)，便于与输入张量相加
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        
        # 返回加上位置编码后的张量
        return x
    
class VitEmbedding(nn.Module):
    """ Image to Patch Embedding 
        Vit的图像块嵌入实现，patch embeddings
        可参照：https://blog.csdn.net/lsb2002/article/details/135320751  or  https://blog.csdn.net/qq_39478403/article/details/118704747
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # (H, W)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # N = (H // P) * (W // P)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape

        assert H == self.img_size[0] and W ==self.img_size[1], \
        f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # (B, C, H, W) -> (B, D, (H//P), (W//P)) -> (B, D, N) -> (B, N, D)
        #   D=embed_dim=768, N=num_patches=(H//P)*(W//P)
        #   torch.flatten(input, start_dim=0, end_dim=-1)  # 形参：展平的起始维度和结束维度    
        x = self.proj(x).flatten(2).transpose(1,2)
        return x
    
# 测试 PositionalEmbedding 的函数
def test_positional_embedding():
    max_len = 5000  # 最大序列长度
    hidden_size = 512  # 嵌入维度
    batch_size = 2
    seq_len = 4
    
    # 随机生成输入数据，形状为 (batch_size, seq_len, hidden_size)
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # 创建 PositionalEmbedding 模块实例
    positional_embedding = PositionalEmbedding(max_len, hidden_size)
    
    # 计算位置嵌入输出
    output = positional_embedding(x)
    
    # 打印输入和输出的形状
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

# 如果此模块是主模块，则运行测试函数
if __name__ == "__main__":
    test_positional_embedding()