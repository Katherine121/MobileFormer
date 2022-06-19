import torch
import torch.nn as nn

from torch.nn import init
import torch.nn.functional as F

from mobile_former.mobile import Mobile, MobileDown
from mobile_former.former import Former
from mobile_former.bridge import Mobile2Former, Former2Mobile


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class BaseBlock(nn.Module):
    def __init__(self, inp, exp, out, se, stride, heads, dim):
        super(BaseBlock, self).__init__()
        if stride == 2:
            self.mobile = MobileDown(3, inp, exp, out, stride, dim)
        else:
            self.mobile = Mobile(3, inp, exp, out, stride, dim)
        self.mobile2former = Mobile2Former(dim=dim, heads=heads, channel=inp)
        self.former = Former(dim=dim)
        self.former2mobile = Former2Mobile(dim=dim, heads=heads, channel=out)

    def forward(self, x, z):
        z_hid = self.mobile2former(x, z)
        z_out = self.former(z_hid)
        x_hid = self.mobile(x, z_out)
        x_out = self.former2mobile(x_hid, z_out)
        return x_out, z_out


class MobileFormer(nn.Module):
    def __init__(self, cfg):
        super(MobileFormer, self).__init__()
        # 初始化6*192token
        self.token = nn.Parameter(nn.Parameter(torch.randn(1, cfg['token'], cfg['embed'])))
        # stem 3 224 224 -> 16 112 112
        self.stem = nn.Sequential(
            nn.Conv2d(3, cfg['stem'], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cfg['stem']),
            hswish(),
        )
        # bneck 先*2后还原，步长为1，组卷积
        self.bneck = nn.Sequential(
            nn.Conv2d(cfg['stem'], cfg['bneck']['e'], kernel_size=3, stride=cfg['bneck']['s'], padding=1, groups=cfg['stem']),
            hswish(),
            nn.Conv2d(cfg['bneck']['e'], cfg['bneck']['o'], kernel_size=1, stride=1),
            nn.BatchNorm2d(cfg['bneck']['o'])
        )

        # body
        self.block = nn.ModuleList()
        for kwargs in cfg['body']:
            # 把{'inp': 12, 'exp': 72, 'out': 16, 'se': None, 'stride': 2, 'heads': 2}和token维度192传进去
            self.block.append(BaseBlock(**kwargs, dim=cfg['embed']))

        inp = cfg['body'][-1]['out']
        exp = cfg['body'][-1]['exp']
        self.conv = nn.Conv2d(inp, exp, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(exp)
        self.avg = nn.AvgPool2d((7, 7))

        self.head = nn.Sequential(
            nn.Linear(exp + cfg['embed'], cfg['fc1']),
            hswish(),
            nn.Linear(cfg['fc1'], cfg['fc2'])
        )

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # batch_size
        b = x.shape[0]
        # 因为最开始初始化的是1*6*192，在0维度重复b次，1维度重复1次，2维度重复1次，就形成了b*6*192
        z = self.token.repeat(b, 1, 1)

        x = self.bneck(self.stem(x))

        for m in self.block:
            x, z = m(x, z)

        # 转成b个平铺一维向量
        x = self.avg(self.bn(self.conv(x))).view(b, -1)
        # 取第一个token
        z = z[:, 0, :].view(b, -1)
        # 最后一个维度拼接
        out = torch.cat((x, z), -1)

        return self.head(out)
