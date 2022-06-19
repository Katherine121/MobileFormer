import torch
from torch import nn
from einops import rearrange


# inputs: x(b c h w) z(b m d)
# output: z(b m d)
class Mobile2Former(nn.Module):
    def __init__(self, dim, heads, channel, dropout=0.3):
        super(Mobile2Former, self).__init__()
        inner_dim = heads * channel
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim)
        self.scale = channel ** -0.5
        self.attend = nn.Softmax(dim=-1)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, z):
        b, m, d = z.shape
        b, c, h, w = x.shape

        # b, m, d -> b, m, head*c -> b, head, m, c
        q = self.to_q(z).view(b, self.heads, m, c)
        # b, c, h, w -> b, 1, h*w, c
        x = x.reshape(b, c, h*w).transpose(1,2).unsqueeze(1)

        # 矩阵相乘 b, head, m, c @ b, 1, c, h*w -> b, head, m, h*w
        dots = q @ x.transpose(2, 3) * self.scale
        attn = self.attend(dots)
        # b, head, m, h*w @ b, 1, h*w, c -> b, head, m, c
        out = attn @ x
        # b, head, m, c -> b, m, head*c
        out = rearrange(out, 'b h m c -> b m (h c)')
        # b, m, head*c -> b, m, d
        return z + self.to_out(out)


# inputs: x(b c h w) z(b m d)
# output: x(b c h w)
class Former2Mobile(nn.Module):
    def __init__(self, dim, heads, channel, dropout=0.3):
        super(Former2Mobile, self).__init__()
        inner_dim = heads * channel
        self.heads = heads
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.scale = channel ** -0.5
        self.attend = nn.Softmax(dim=-1)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, channel),
            nn.Dropout(dropout)
        )

    def forward(self, x, z):
        b, m, d = z.shape
        b, c, h, w = x.shape
        # b,c,h*w -> b,1,h*w,c
        q = x.reshape(b, c, h*w).transpose(1,2).unsqueeze(1)
        # b,m,d -> b,m,head*c -> b,head,m,c
        k = self.to_k(z).view(b, self.heads, m, c)
        v = self.to_v(z).view(b, self.heads, m, c)
        # b,1,h*w,c @ b,head,c,m -> b,head,h*w,m
        dots = q @ k.transpose(2, 3) * self.scale
        attn = self.attend(dots)
        # b,head,h*w,m @ b,head,m,c -> b,head,h*w,c
        out = attn @ v
        # b,head,h*w,c -> b,h*w,head*c
        out = rearrange(out, 'b h l c -> b l (h c)')
        # b,h*w,head*c -> b,h*w,c
        out = self.to_out(out)
        out = out.view(b, c, h, w)
        return x + out
