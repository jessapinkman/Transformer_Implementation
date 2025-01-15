import numpy as np
import torch.nn.functional as F
import torch
from torch import nn

# from mmcv.runner import auto_fp16
import math
from torch.nn.parallel import DistributedDataParallel as DDP


def split_last(x, shape):
    """split the last dimension to given shape"""
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    """merge the last n_dims to a dimension"""
    s = x.size()
    assert 1 < n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1


def repeat_kv(x, n_rep: int):
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class MultiHeadedCacheAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""

    def __init__(self, dim, num_heads, num_kv_heads=None):
        """
        通过缓存并拼接历史 Key 和 Value，大幅提升了推理阶段的效率.再普通attention中推理时间随序列长度增长，
        cache后推理时间仅与当前query的长度成正比，比较适合自回归模型，但是需要增加额外内存占用并且再mask中也需要考虑到历史key、value
        Initialize the Attention module.

        n_kv_heads (int): Number of key and value heads.
        n_local_heads (int): Number of local query heads.
        n_local_kv_heads (int): Number of local key and value heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (ColumnParallelLinear): Linear transformation for queries.
        wk (ColumnParallelLinear): Linear transformation for keys.
        wv (ColumnParallelLinear): Linear transformation for values.
        wo (RowParallelLinear): Linear transformation for output.
        cache_k (torch.Tensor): Cached keys for attention.
        cache_v (torch.Tensor): Cached values for attention.
        """
        super(MultiHeadedCacheAttention, self).__init__()

        self.hidden_size = dim # num of dimension
        self.n_heads = num_heads # num of query head
        self.n_kv_heads = num_kv_heads if num_kv_heads else num_heads # num of kv heads
        self.head_dim = dim // num_heads # num od dimension for each head
        self.n_rep = self.n_heads // self.n_kv_heads # 当 num_heads > num_kv_heads 时，表示需要对 Key 和 Value 的重复次数，用于支持查询更多头的场景
        self.proj_q = nn.Linear(dim, self.n_heads * self.head_dim)
        self.proj_k = nn.Linear(dim, self.n_kv_heads * self.head_dim)
        self.proj_v = nn.Linear(dim, self.n_kv_heads * self.head_dim)
        self.proj_o = nn.Linear(self.n_heads * self.head_dim, dim)
        self.key_cache = None
        self.value_cache = None
        # initially in training mode
        self.is_training = True
        self.data_id = -1

    def set_mode(self, is_training):
        self.is_training = is_training
        if is_training:
            self.clear_cache()

    def set_data_id(self, data_id):
        self.data_id = data_id

    def clear_cache(self, data_id=None):
        """如果传入data_id，则仅清空对应数据的缓存"""
        if data_id is not None:
            self.key_cache[data_id] = None
            self.value_cache[data_id] = None
        else:
            self.key_cache = {self.data_id: None}
            self.value_cache = {self.data_id: None}

    def forward(self, x, y, mask_x, mask_y, attn_mask):
        """xy for agent to agent or agent to map or map to map, for crossattention"""
        bsz, seqlen, _ = x.shape
        query, key, value = self.proj_q(x), self.proj_k(y), self.proj_v(y)
        query = query.view(bsz, seqlen, self.n_heads, self.head_dim)
        key = key.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        value = value.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # repeat k/v heads if n_kv_heads < n_heads, used for match query n_heads
        key = repeat_kv(key, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        value = repeat_kv(value, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        query, key, value = (x.transpose(1, 2) for x in [query, key, value])  # (bs, n_local_heads, seqlen, head_dim)

        if self.is_training:
            # Training mode: No caching, fully parallel
            self.key_cache = {self.data_id: key}
            self.value_cache = {self.data_id: value}
        else:
            # Inference mode: Use caching
            if self.data_id not in self.key_cache.keys():
                self.key_cache[self.data_id] = key
                self.value_cache[self.data_id] = value
            else: # 对于后续时间步，将新生成的 Key 和 Value 拼接到缓存中，避免重复计算
                self.key_cache[self.data_id] = torch.cat([self.key_cache[self.data_id], key], dim=2) # (bs, n_local_heads, seqlen, head_dim)
                self.value_cache[self.data_id] = torch.cat([self.value_cache[self.data_id], value], dim=2)

        scores = torch.matmul(query, self.key_cache[self.data_id].transpose(-2, -1)) / self.key_cache[self.data_id].size(-1) ** 0.5
        if mask_x is not None and mask_y is not None:
            mask_x = mask_x[:, None, :, None]  # (B, Sx) -> (B, 1, Sx, 1)
            mask_y = mask_y[:, None, None, :]  # (B, Sy) -> (B, 1, 1, Sy)
            mask = mask_x * mask_y
            scores -= 100000.0 * (1.0 - mask) # 被mask的地方就是1 * 一个很大的值，score减去就很小，再过softmax会趋于0

        if attn_mask is not None:
            mask = attn_mask[:, None]  # (B, Sx, Sy) -> (B, 1, Sx, Sy)
            scores -= 100000.0 * (1.0 - mask)

        scores = F.softmax(scores, dim=-1)
        h = torch.matmul(scores, self.value_cache[self.data_id])
        h = h.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        h = self.proj_o(h)

        return h
    


class ModelManager:
    """
    model.load_state_dict(checkpoint["state_dict"])
    # model = torch.nn.DataParallel(model)
    model = model.to(device).eval()
    manager = ModelManager(model)
    manager.set_mode(False)
    manager.clear_cache() # 每推理完一次要clear
    """
    def __init__(self, model):
        self.model = model

    def set_mode(self, is_training):
        self._set_mode_recursive(self.model, is_training)

    def clear_cache(self):
        self._clear_cache_recursive(self.model)

    def set_data_id(self, data_id):
        self._set_data_id_recursive(self.model, data_id)

    def _set_mode_recursive(self, module, is_training):
        if isinstance(module, DDP):
            module = module.module  # 访问底层模型
        if isinstance(module, MultiHeadedCacheAttention):
            module.set_mode(is_training)
        # if isinstance(module, MultiHeadedFixedKVAttention):
        #     module.set_mode(is_training)
        else:
            for child in module.children():
                self._set_mode_recursive(child, is_training)

    def _clear_cache_recursive(self, module):
        if isinstance(module, DDP):
            module = module.module  # 访问底层模型
        if isinstance(module, MultiHeadedCacheAttention):
            module.clear_cache()
        # if isinstance(module, MultiHeadedFixedKVAttention):
        #     module.clear_cache()
        else:
            for child in module.children():
                self._clear_cache_recursive(child)

    def _set_data_id_recursive(self, module, data_id):
        if isinstance(module, DDP):
            module = module.module  # 访问底层模型
        if isinstance(module, MultiHeadedCacheAttention):
            module.set_data_id(data_id)
        else:
            for child in module.children():
                self._set_data_id_recursive(child, data_id)

