import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import exp

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1) adapted for MST"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss

class SSIMLoss(nn.Module):
    """SSIM Loss for hyperspectral images"""

    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        ssim_value = self._ssim(img1, img2, window, self.window_size, channel, self.size_average)
        return 1 - ssim_value

class SpectralAngleLoss(nn.Module):
    """Spectral Angle Mapper (SAM) Loss for hyperspectral images"""

    def __init__(self, reduction='mean'):
        super(SpectralAngleLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        """
        Calculate SAM loss between input and target
        Args:
            input: [batch_size, channels, height, width]
            target: [batch_size, channels, height, width]
        """
        # Flatten spatial dimensions
        input_flat = input.view(input.shape[0], input.shape[1], -1)   # [B, C, H*W]
        target_flat = target.view(target.shape[0], target.shape[1], -1) # [B, C, H*W]

        # Calculate dot product along spectral dimension
        dot_product = torch.sum(input_flat * target_flat, dim=1)  # [B, H*W]

        # Calculate norms
        input_norm = torch.norm(input_flat, dim=1)   # [B, H*W]
        target_norm = torch.norm(target_flat, dim=1) # [B, H*W]

        # Avoid division by zero
        norm_product = input_norm * target_norm
        norm_product = torch.clamp(norm_product, min=1e-8)

        # Calculate cosine of angle
        cos_angle = dot_product / norm_product
        cos_angle = torch.clamp(cos_angle, -1, 1)  # Ensure valid range for acos

        # Calculate spectral angle
        angle = torch.acos(cos_angle)  # [B, H*W]

        if self.reduction == 'mean':
            return torch.mean(angle)
        elif self.reduction == 'sum':
            return torch.sum(angle)
        else:
            return angle

class HybridLoss(nn.Module):
    """Hybrid loss combining multiple loss functions for HSI reconstruction"""

    def __init__(self, l1_weight=1.0, charbonnier_weight=1.0, ssim_weight=1.0, sam_weight=0.1):
        super(HybridLoss, self).__init__()
        self.l1_weight = l1_weight
        self.charbonnier_weight = charbonnier_weight
        self.ssim_weight = ssim_weight
        self.sam_weight = sam_weight

        self.l1_loss = nn.L1Loss()
        self.charbonnier_loss = CharbonnierLoss()
        self.ssim_loss = SSIMLoss()
        self.sam_loss = SpectralAngleLoss()

    def forward(self, input, target):
        """
        Calculate hybrid loss
        Args:
            input: Reconstructed HSI [batch_size, channels, height, width]
            target: Ground truth HSI [batch_size, channels, height, width]
        """
        l1 = self.l1_loss(input, target)
        charbonnier = self.charbonnier_loss(input, target)
        ssim = self.ssim_loss(input, target)
        sam = self.sam_loss(input, target)

        total_loss = (self.l1_weight * l1 +
                     self.charbonnier_weight * charbonnier +
                     self.ssim_weight * ssim +
                     self.sam_weight * sam)

        return {
            'total': total_loss,
            'l1': l1,
            'charbonnier': charbonnier,
            'ssim': ssim,
            'sam': sam
        }

class PerceptualLoss(nn.Module):
    """Perceptual loss using pre-trained features (simplified version for HSI)"""

    def __init__(self, feature_layers=[2, 7, 12]):
        super(PerceptualLoss, self).__init__()
        # Simple conv layers to extract features
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(84, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.feature_layers = feature_layers
        self.mse_loss = nn.MSELoss()

    def forward(self, input, target):
        """
        Calculate perceptual loss using feature differences
        """
        input_features = []
        target_features = []

        x_input = input
        x_target = target

        for i, layer in enumerate(self.feature_extractor):
            x_input = layer(x_input)
            x_target = layer(x_target)

            if i in self.feature_layers:
                input_features.append(x_input)
                target_features.append(x_target)

        loss = 0
        for feat_input, feat_target in zip(input_features, target_features):
            loss += self.mse_loss(feat_input, feat_target)

        return loss / len(input_features)

def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculate PSNR between two images
    Args:
        img1, img2: Input tensors
        max_val: Maximum possible pixel value
    Returns:
        PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return torch.tensor(100.0)
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def calculate_ergas(img_true, img_pred, r=1, ws=1):
    """
    Calculate ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse)
    Args:
        img_true: Ground truth image [B, C, H, W]
        img_pred: Predicted image [B, C, H, W]
        r: Ratio between pixel sizes
        ws: Water surface flag
    Returns:
        ERGAS value
    """
    if len(img_true.shape) == 4:
        img_true = img_true.view(img_true.shape[0], img_true.shape[1], -1)
        img_pred = img_pred.view(img_pred.shape[0], img_pred.shape[1], -1)

    mean_true = torch.mean(img_true, dim=-1)  # [B, C]
    mse = torch.mean((img_true - img_pred) ** 2, dim=-1)  # [B, C]

    ergas_bands = mse / (mean_true ** 2 + 1e-8)
    ergas = 100 * r * torch.sqrt(torch.mean(ergas_bands))

    return ergas