import torch
from torch import nn
from MultiHeadAttention import MultiHeadAttention
from Embedding import PositionalEmbedding, TokenEmbedding, VitEmbedding
from .EncoderLayer import EncoderLayer


class VitEncoderLayer(nn.Module):
    '''比一般的tf encoder layer多了一个patch embedding，并且在编码器后多接了一个mlp做分类头
       留个坑，等真正需要用了再来完善下面的手撕
    '''
    def __init__(self, hidden_size, num_heads, ff_size, num_features, num_classes=512):
        super().__init__()
        ### 随机初始化
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))  # shape = (1, 1, D)

        self.patch_embed = VitEmbedding(embed_dim=hidden_size)
        self.pos_embed = PositionalEmbedding(hidden_size=hidden_size)
        
        ### 分类头 (Classifier head)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        
    def forward(self, x):
        B = x.shape[0] 
        x = self.patch_embed(x)  # x.shape = (B, N, D)
        
        # 可学习嵌入 - 用于分类
        cls_tokens = self.cls_token.expand(B, -1, -1)  # shape = (B, 1, D)
        
        # 按元素相加 附带 Position Embeddings
        x = x + self.pos_embed  # shape = (B, N, D)
        
        # 按通道拼接 获取 N+1 维 Embeddings
        x = torch.cat((cls_tokens, x), dim=1)  # shape = (B, N+1, D)

        return x
