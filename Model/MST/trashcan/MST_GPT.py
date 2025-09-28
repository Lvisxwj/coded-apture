import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out

# ---------- init helpers ----------
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "Values may be incorrect.", stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    denom = fan_in if mode == 'fan_in' else fan_out if mode == 'fan_out' else (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")

def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')

# ---------- small utils ----------
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GELU(nn.Module):
    def forward(self, x): return F.gelu(x)

def shift_back(inputs, step=2):
    """If col > row (如 256x310) 按通道索引做位移并裁到正方。否则直接返回。"""
    bs, nC, row, col = inputs.shape
    if col == row or step is None:
        return inputs
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = inputs[:, i, :, int(step * i):int(step * i) + out_col]
    return inputs[:, :, :, :out_col]

# ---------- mask guidance ----------
class MaskGuidedMechanism(nn.Module):
    """接收已对齐到 dim 通道的 mask，做深度可分 + 门控 + 可选 shift_back。"""
    def __init__(self, dim, step=None):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, bias=True, groups=dim)
        self.step = step
    def forward(self, mask_shift):
        mask_shift = self.conv1(mask_shift)
        attn_map = torch.sigmoid(self.depth_conv(self.conv2(mask_shift)))
        mask_shift = mask_shift * attn_map + mask_shift
        mask_emb = shift_back(mask_shift, step=self.step)
        return mask_emb

# ---------- attention blocks ----------
class MS_MSA(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, step=None):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
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
        self.mm = MaskGuidedMechanism(dim, step=step)
        self.dim = dim

    def forward(self, x_in, mask_emb):
        """
        x_in: [b, c, h, w]
        mask_emb: [b, c, h, w] —— 已用 MaskAdapter 投影到 c=dim
        """
        b, c, h, w = x_in.shape
        x = x_in.permute(0, 2, 3, 1).reshape(b, h * w, c)  # [b, hw, c]
        q_inp = self.to_q(x); k_inp = self.to_k(x); v_inp = self.to_v(x)

        # mask guidance
        mask_attn = self.mm(mask_emb).permute(0, 2, 3, 1)  # [b, h, w, c]
        if b != 0:
            mask_attn = mask_attn.expand(b, h, w, c)

        # heads
        q, k, v, m = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                         (q_inp, k_inp, v_inp, mask_attn.reshape(b, -1, c)))
        v = v * m
        q = F.normalize(q.transpose(-2, -1), dim=-1, p=2)
        k = F.normalize(k.transpose(-2, -1), dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # [b, heads, d, hw]
        x = x.permute(0, 3, 1, 2).reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c).permute(0, 3, 1, 2)

        out_p = self.pos_emb(v_inp.view(b, h, w, c).permute(0, 3, 1, 2))
        return out_c + out_p

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
    def forward(self, x): return self.net(x)

class MSAB(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2, step=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads, step=step),
                PreNorm(dim, lambda t: FeedForward(dim)(t))
            ])
            for _ in range(num_blocks)
        ])
    def forward(self, x, mask_emb):
        for (attn, ff) in self.blocks:
            x = attn(x, mask_emb=mask_emb) + x
            x = ff(x) + x
        return x

# ---------- MST with mask adapters ----------
class MST(nn.Module):
    def __init__(self,
                 dim=64, stage=3, num_blocks=[2, 2, 2],
                 in_channels=1, out_channels=3,
                 mask_in_channels=1, step=None):
        """
        step: 与 HSI/CASSI 相关的横向位移参数；普通 256x256 任务可设 None（不做 shift_back）
        """
        super(MST, self).__init__()
        self.dim = dim
        self.stage = stage
        self.step = step

        # 输入/输出投影
        self.embedding = nn.Conv2d(in_channels, self.dim, 3, 1, 1, bias=False)
        self.mapping   = nn.Conv2d(self.dim, out_channels, 3, 1, 1, bias=False)

        # 编码器
        self.encoder_layers = nn.ModuleList([])
        self.mask_down = nn.ModuleList([])       # 仅做空间下采样
        self.mask_adapt_enc = nn.ModuleList([])  # 通道适配：mask -> dim_stage

        dim_stage = dim
        m_in = mask_in_channels
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                MSAB(dim=dim_stage, num_blocks=num_blocks[i],
                     dim_head=dim, heads=max(1, dim_stage // dim), step=step),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),  # FeaDown
            ]))
            self.mask_down.append(nn.AvgPool2d(kernel_size=2))
            self.mask_adapt_enc.append(nn.Conv2d(m_in, dim_stage, kernel_size=1, bias=True))
            dim_stage *= 2

        # 瓶颈
        self.bottleneck = MSAB(dim=dim_stage, dim_head=dim,
                               heads=max(1, dim_stage // dim), num_blocks=num_blocks[-1], step=step)
        self.mask_adapt_bot = nn.Conv2d(m_in, dim_stage, kernel_size=1, bias=True)

        # 解码器
        self.decoder_layers = nn.ModuleList([])
        self.mask_adapt_dec = nn.ModuleList([])

        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),  # Fusion
                MSAB(dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i],
                     dim_head=dim, heads=max(1, (dim_stage // 2) // dim), step=step),
            ]))
            self.mask_adapt_dec.append(nn.Conv2d(m_in, dim_stage // 2, kernel_size=1, bias=True))
            dim_stage //= 2

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, mask=None):
        """
        x:    [b, in_channels, h, w]      e.g. [b,1,256,256]
        mask: [b, mask_in_channels, h, w] e.g. [b,1,256,256]（可为编码模板/先验；不传则全零）
        """
        if mask is None:
            mask = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)

        # Embedding
        fea = self.lrelu(self.embedding(x))

        # Encoder
        fea_skips = []
        mask_skips = []
        cur_mask = mask
        for (msab, fea_down), mask_pool, mask_adapt in zip(self.encoder_layers, self.mask_down, self.mask_adapt_enc):
            mask_emb = mask_adapt(cur_mask)          # [b, dim_stage, h, w]
            fea = msab(fea, mask_emb)                # 注意：MSAB 里认为 mask 与 fea 同通道
            fea_skips.append(fea)
            mask_skips.append(cur_mask)              # 保存未通道投影的 mask（只做空间）
            fea = fea_down(fea)                      # 下采样特征
            cur_mask = mask_pool(cur_mask)           # 下采样 mask（空间）

        # Bottleneck
        fea = self.bottleneck(fea, self.mask_adapt_bot(cur_mask))

        # Decoder
        for i, (fea_up, fusion, msab) in enumerate(self.decoder_layers):
            fea = fea_up(fea)
            fea = fusion(torch.cat([fea, fea_skips[self.stage - 1 - i]], dim=1))
            cur_mask = mask_skips[self.stage - 1 - i]
            mask_emb = self.mask_adapt_dec[i](cur_mask)
            fea = msab(fea, mask_emb)

        # Output
        out = self.mapping(fea) + (x if self.mapping.out_channels == x.size(1) else 0)
        return out
