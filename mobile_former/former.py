import torch
from torch import nn
from einops import rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.3):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            # 高斯误差线性单元激活函数，常用于BERT
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, z):
        return self.net(z)


class Attention(nn.Module):
    def __init__(self, dim=192, heads=2, dim_head=32, dropout=0.3):
        super(Attention, self).__init__()
        inner_dim = heads * dim_head  # head数量和每个head的维度

        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, z):
        # 先经过全连接层获得qkv，然后分割
        # b,6,192 -> b,6,64 + b,6,64 + b,6,64
        qkv = self.to_qkv(z).chunk(3, dim=-1)
        # q: b,6,64 -> b,2,6,32  2个head，每个head维度32
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
                      qkv)

        # dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # b,2,6,32 @ b,2,6,32 -> b,2,6,6
        # b,2,6,32 @ b,2,32,6 -> b,2,6,6
        dots = q @ k.transpose(2, 3) * self.scale
        attn = self.attend(dots)

        # 每个token经过每个head的attention后的输出
        # out = einsum('b h i j, b h j d -> b h i d', attn, v)  # atten@v b,2,6,6 @ b,2,6,32 -> b,2,6,32
        # b,2,6,6 @ b,2,6,32 -> b,2,6,32
        out = attn @ v

        # b,2,6,32 -> b,6,64
        out = rearrange(out, 'b h n d -> b n (h d)')  # 合并所有head的输出b,6,64
        return self.to_out(out)


# inputs: n m d
# output: n m d
class Former(nn.Module):
    def __init__(self, dim, depth=1, heads=2, dim_head=32, dropout=0.3):
        super(Former, self).__init__()
        # 2 instead of 4
        mlp_dim = dim * 2
        self.layers = nn.ModuleList([])
        self.layers.append(nn.ModuleList([
            # 先bn，后多头注意力，再前馈
            PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
            # 先*2后还原
            PreNorm(dim, FeedForward(dim, mlp_dim=mlp_dim, dropout=dropout))
        ]))

    def forward(self, z):
        attn = self.layers[0][0]
        ff = self.layers[0][1]
        # 残差连接
        z = attn(z) + z
        z = ff(z) + z
        return z
