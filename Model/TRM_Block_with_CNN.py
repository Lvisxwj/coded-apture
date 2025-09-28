import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from collections import OrderedDict
from einops.layers.torch import Rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)

def shift_back(inputs,step=2):          # input [bs,28,256,310]  output [bs, 28, 256, 256]
    [bs, nC, row, col] = inputs.shape
    down_sample = 256//row
    step = float(step)/float(down_sample*down_sample)
    out_col = row
    for i in range(nC):
        inputs[:,i,:,:out_col] = \
            inputs[:,i,:,int(step*i):int(step*i)+out_col]
    return inputs[:, :, :, :out_col]

class MS_MSA_C(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=16,
            method='dw_bn'
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head

        # conv
        self.conv_proj_q = self._build_projection(
            dim, dim_head * heads, 3, 1,
            1, 'linear' if method == 'avg' else method
        )
        self.conv_proj_k = self._build_projection(
            dim, dim_head * heads, 3, 1,
            1, method
        )
        self.conv_proj_v = self._build_projection(
            dim, dim_head * heads, 3, 1,
            1, method
        )
        # 线性
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)

        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def _build_projection(self,
                      dim_in,
                      dim_out,
                      kernel_size,
                      padding,
                      stride,
                      method):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim_in,
                    dim_out,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                ('bn', nn.BatchNorm2d(dim_out)),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj
    
    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        # 如要使用msa，则将最前面的permute删去，最后的也不要
        # x_in = x_in.permute(0, 2, 3, 1)
        # print(x_in.shape) # torch.Size([1, 32, 128, 32])
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        # print(x.shape) # torch.Size([1, 4096, 32])
        """
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        """
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        # print(x.shape) # torch.Size([16, 64, 64, 64])

        if self.conv_proj_q is not None:
            q_inp = self.conv_proj_q(x)
        else:
            q_inp = rearrange(x, 'b c h w -> b (h w) c')
        # print(q_inp.shape)
        if self.conv_proj_k is not None:
            k_inp = self.conv_proj_k(x)
        else:
            k_inp = rearrange(x, 'b c h w -> b (h w) c')
        if self.conv_proj_v is not None:
            v_inp = self.conv_proj_v(x)
        else:
            v_inp = rearrange(x, 'b c h w -> b (h w) c')
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        # print(attn.shape)
        x = attn @ v   # b,heads,d,hw
        # print(x.shape)
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        # print(x.shape)
        out_c = self.proj(x).view(b, h, w, c)
        # print(out_c.shape)
        out_p = self.pos_emb(self.proj(v_inp).reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # print(out_p.shape)
        out = out_c + out_p
        # print(out.shape)
        # out = out.permute(0, 3, 1, 2)
        # print(out.shape) # torch.Size([1, 48, 48, 128])

        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class MSABC(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=16,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA_C(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        # print(x.shape)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out
"""
class DLGD(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.DL = nn.Sequential(
            PWDWPWConv(29, 28, opt.bias, act_fn_name=opt.act_fn_name),
            PWDWPWConv(28, 28, opt.bias, act_fn_name=opt.act_fn_name),
        )
        self.r = nn.Parameter(torch.Tensor([0.5]))


    def forward(self, y, xk_1, Phi):

        y    : (B, 1, 256, 310)
        xk_1 : (B, 28, 256, 310)
        phi  : (B, 28, 256, 310)

        DL_Phi = self.DL(torch.cat([y.unsqueeze(1), Phi], axis=1))
        Phi = DL_Phi + Phi
        phi = A(xk_1, Phi) # (B, 256, 310)
        phixsy = phi - y
        phit = At(phixsy, Phi)
        r = self.r
        vk = xk_1 - r * phit

        return vk
"""
