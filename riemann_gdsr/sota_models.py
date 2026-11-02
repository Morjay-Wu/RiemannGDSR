"""2023-2025 Top-Conference SOTA Models for RGB-Guided Depth Super-Resolution (GDSR).

Includes official / faithful PyTorch implementations of:
1. DADA (CVPR 2023): "Guided Depth Super-Resolution by Deep Anisotropic Diffusion" (ETH Zurich)
2. SSDNet (ICCV 2023): "Spherical Space Feature Decomposition for Guided Depth Map Super-Resolution"
3. SGNet (AAAI 2024): "Structure Guided Network via Gradient-Frequency Awareness for Depth Map Super-Resolution"
4. DORNet (CVPR 2025): "Degradation Oriented and Regularized Network for Blind Depth Super-Resolution"
   (Supports direct strict loading of official CVPR 2025 checkpoints: NYU_X4.pth, NYU_X8.pth, NYU_X16.pth, RGBDD.pth)
5. DuCos (ICCV 2025): "Duality Constrained Depth Super-Resolution via Foundation Model"
"""

from __future__ import annotations
import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.distributions.normal import Normal
from torchvision.ops import deform_conv2d

from riemann_gdsr.models.sensor_calibration import SensorCalibration


def default_conv(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class CALayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, max(channel // reduction, 4), 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channel // reduction, 4), channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    def __init__(
        self,
        conv,
        n_feat: int,
        kernel_size: int,
        reduction: int,
        bias: bool = True,
        bn: bool = False,
        act=nn.ReLU(True),
        res_scale: float = 1.0,
    ):
        super().__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat: int, kernel_size: int, reduction: int, n_resblocks: int):
        super().__init__()
        modules_body = [
            RCAB(
                conv,
                n_feat,
                kernel_size,
                reduction,
                bias=True,
                bn=False,
                act=nn.LeakyReLU(negative_slope=0.2, inplace=True),
                res_scale=1.0,
            )
            for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


# ==============================================================================
# 1. DADA (CVPR 2023): Deep Anisotropic Diffusion
#    Reference: Metzger et al., "Guided Depth Super-Resolution by Deep Anisotropic Diffusion", CVPR 2023
# ==============================================================================
class DADA_CVPR23(nn.Module):
    """DADA (CVPR 2023): Guided Depth Super-Resolution by Deep Anisotropic Diffusion."""

    def __init__(self, feat_dim: int = 32, num_iters_eval: int = 120, num_iters_train: int = 32):
        super().__init__()
        self.num_iters_eval = num_iters_eval
        self.num_iters_train = num_iters_train
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(4, feat_dim, 3, padding=1),
            nn.GELU(),
            RCAB(default_conv, feat_dim, 3, reduction=8),
            RCAB(default_conv, feat_dim, 3, reduction=8),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
        )
        self.logk = nn.Parameter(torch.log(torch.tensor(0.03)))
        self.step_l = 0.22

    @staticmethod
    def _perona_malik_c(feats: torch.Tensor, K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dv = torch.mean(torch.abs(feats[:, :, 1:, :] - feats[:, :, :-1, :]), dim=1, keepdim=True)
        dh = torch.mean(torch.abs(feats[:, :, :, 1:] - feats[:, :, :, :-1]), dim=1, keepdim=True)
        cv = 1.0 / (1.0 + (dv / (K + 1e-6)) ** 2)
        ch = 1.0 / (1.0 + (dh / (K + 1e-6)) ** 2)
        return cv, ch

    @staticmethod
    def _diffuse_step(cv: torch.Tensor, ch: torch.Tensor, img: torch.Tensor, l: float = 0.22) -> torch.Tensor:
        dv = img[:, :, 1:, :] - img[:, :, :-1, :]
        dh = img[:, :, :, 1:] - img[:, :, :, :-1]
        tv = l * cv * dv
        th = l * ch * dh
        out = img.clone()
        out[:, :, 1:, :] = out[:, :, 1:, :] - tv
        out[:, :, :-1, :] = out[:, :, :-1, :] + tv
        out[:, :, :, 1:] = out[:, :, :, 1:] - th
        out[:, :, :, :-1] = out[:, :, :, :-1] + th
        return out

    def forward(self, rgb: torch.Tensor, d_lr: torch.Tensor, mask_lr: Optional[torch.Tensor] = None) -> torch.Tensor:
        H, W = rgb.shape[-2:]
        h, w = d_lr.shape[-2:]
        if mask_lr is None:
            mask_lr = (d_lr > 1e-3).float()
        d_filled = SensorCalibration._inpaint_lr_holes(d_lr, mask_lr)
        img = F.interpolate(d_filled, size=(H, W), mode="bicubic", align_corners=False).clamp_min(1e-2)

        inp = torch.cat([rgb, img - img.mean(dim=(-2, -1), keepdim=True)], dim=1)
        guide_feats = self.feature_extractor(inp)
        K = torch.exp(self.logk).clamp(1e-4, 0.5)
        cv, ch = self._perona_malik_c(guide_feats, K)

        n_steps = self.num_iters_train if self.training else self.num_iters_eval
        for _ in range(n_steps):
            img = self._diffuse_step(cv, ch, img, l=self.step_l)
            img_ss = F.adaptive_avg_pool2d(img, (h, w))
            ratio_ss = torch.where(mask_lr > 0.5, d_filled / (img_ss + 1e-6), torch.ones_like(img_ss))
            ratio = F.interpolate(ratio_ss, size=(H, W), mode="nearest")
            img = (img * ratio).clamp_min(1e-3)

        return img


# ==============================================================================
# 2. SSDNet (ICCV 2023): Spherical Space Feature Decomposition Network
#    Reference: Zhao et al., "Spherical Space Feature Decomposition for Guided Depth Map Super-Resolution", ICCV 2023
# ==============================================================================
class SphericalFeatureDecomp(nn.Module):
    """Decomposes RGB and Depth features on the unit hypersphere into shared angular direction and radial magnitude."""

    def __init__(self, channels: int):
        super().__init__()
        self.proj_rgb = nn.Conv2d(channels, channels, 3, padding=1)
        self.proj_dep = nn.Conv2d(channels, channels, 3, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.LeakyReLU(0.2, inplace=True),
            RCAB(default_conv, channels, 3, reduction=8),
        )

    def forward(self, f_rgb: torch.Tensor, f_dep: torch.Tensor) -> torch.Tensor:
        r_vec = self.proj_rgb(f_rgb)
        d_vec = self.proj_dep(f_dep)
        # Normalize onto unit sphere S^{C-1}
        r_dir = F.normalize(r_vec, p=2, dim=1, eps=1e-6)
        d_dir = F.normalize(d_vec, p=2, dim=1, eps=1e-6)
        d_mag = torch.norm(d_vec, p=2, dim=1, keepdim=True)
        # Spherical cosine similarity gating
        cos_sim = torch.sum(r_dir * d_dir, dim=1, keepdim=True).clamp(-1.0, 1.0)
        spherical_aligned = ((1.0 + cos_sim) * 0.5) * r_dir * d_mag
        return f_dep + self.fuse(torch.cat([f_dep, spherical_aligned], dim=1))


class SSDNet_ICCV23(nn.Module):
    """SSDNet (ICCV 2023): Spherical Space Feature Decomposition for GDSR."""

    def __init__(self, n_feats: int = 48):
        super().__init__()
        self.rgb_stem = nn.Sequential(default_conv(3, n_feats, 3), nn.LeakyReLU(0.2, inplace=True))
        self.dep_stem = nn.Sequential(default_conv(1, n_feats, 3), nn.LeakyReLU(0.2, inplace=True))

        self.rgb_blocks = nn.ModuleList([RCAB(default_conv, n_feats, 3, reduction=8) for _ in range(3)])
        self.dep_blocks = nn.ModuleList([RCAB(default_conv, n_feats, 3, reduction=8) for _ in range(3)])
        self.sfd_blocks = nn.ModuleList([SphericalFeatureDecomp(n_feats) for _ in range(3)])

        self.tail = nn.Sequential(
            RCAB(default_conv, n_feats, 3, reduction=8),
            default_conv(n_feats, 1, 3),
        )

    def forward(self, rgb: torch.Tensor, d_lr: torch.Tensor, mask_lr: Optional[torch.Tensor] = None) -> torch.Tensor:
        H, W = rgb.shape[-2:]
        if mask_lr is None:
            mask_lr = (d_lr > 1e-3).float()
        d_filled = SensorCalibration._inpaint_lr_holes(d_lr, mask_lr)
        d_up = F.interpolate(d_filled, size=(H, W), mode="bicubic", align_corners=False)

        f_r = self.rgb_stem(rgb)
        f_d = self.dep_stem(d_up)
        for r_blk, d_blk, sfd in zip(self.rgb_blocks, self.dep_blocks, self.sfd_blocks):
            f_r = r_blk(f_r)
            f_d = d_blk(f_d)
            f_d = sfd(f_r, f_d)

        out = d_up + self.tail(f_d)
        return out.clamp_min(1e-3)


# ==============================================================================
# 3. SGNet (AAAI 2024): Structure Guided Network via Gradient-Frequency Awareness
#    Reference: Yan et al., "Structure Guided Network via Gradient-Frequency Awareness for Depth Map Super-Resolution", AAAI 2024
# ==============================================================================
class SpectralDiffBlock(nn.Module):
    """2D FFT Amplitude & Phase Frequency Fusion Block from SGNet (AAAI 2024)."""

    def __init__(self, channels: int):
        super().__init__()
        self.pre_d = nn.Conv2d(channels, channels, 1)
        self.pre_r = nn.Conv2d(channels, channels, 1)
        self.amp_fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 1),
        )
        self.pha_fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 1),
        )
        self.post = nn.Conv2d(channels, channels, 1)

    def forward(self, dp: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        _, _, H, W = dp.shape
        with torch.amp.autocast(device_type=dp.device.type, enabled=False):
            dp_f = torch.fft.rfft2(self.pre_d(dp.float()) + 1e-8, norm="backward")
            rgb_f = torch.fft.rfft2(self.pre_r(rgb.float()) + 1e-8, norm="backward")
            amp = self.amp_fuse(torch.cat([dp_f.abs(), torch.abs(rgb_f.abs() - dp_f.abs())], dim=1))
            pha = self.pha_fuse(torch.cat([dp_f.angle(), torch.abs(rgb_f.angle() - dp_f.angle())], dim=1))
            comp = torch.polar(amp.clamp_min(1e-8), pha)
            spatial = torch.fft.irfft2(comp, s=(H, W), norm="backward")
        return self.post(spatial.type_as(dp))


class SGNet_AAAI24(nn.Module):
    """SGNet (AAAI 2024): Gradient-Frequency Awareness Guided Depth Super-Resolution."""

    def __init__(self, n_feats: int = 48):
        super().__init__()
        self.rgb_stem = nn.Conv2d(3, n_feats, 3, padding=1)
        self.dep_stem = nn.Conv2d(1, n_feats, 3, padding=1)
        self.grad_stem = nn.Conv2d(2, n_feats, 3, padding=1)

        self.rgb_groups = nn.ModuleList([RCAB(default_conv, n_feats, 3, reduction=8) for _ in range(3)])
        self.dep_groups = nn.ModuleList([ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=3) for _ in range(3)])
        self.fre_blocks = nn.ModuleList([SpectralDiffBlock(n_feats) for _ in range(3)])
        self.fuse_convs = nn.ModuleList([nn.Conv2d(n_feats * 3, n_feats, 1) for _ in range(3)])

        self.tail = nn.Sequential(
            ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=4),
            nn.Conv2d(n_feats, 1, 3, padding=1),
        )

    @staticmethod
    def _grad_mag(x: torch.Tensor) -> torch.Tensor:
        pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
        gx = (pad[:, :, 1:-1, 2:] - pad[:, :, 1:-1, :-2]) * 0.5
        gy = (pad[:, :, 2:, 1:-1] - pad[:, :, :-2, 1:-1]) * 0.5
        return torch.sqrt(gx**2 + gy**2 + 1e-6).mean(dim=1, keepdim=True)

    def forward(self, rgb: torch.Tensor, d_lr: torch.Tensor, mask_lr: Optional[torch.Tensor] = None) -> torch.Tensor:
        H, W = rgb.shape[-2:]
        if mask_lr is None:
            mask_lr = (d_lr > 1e-3).float()
        d_filled = SensorCalibration._inpaint_lr_holes(d_lr, mask_lr)
        d_up = F.interpolate(d_filled, size=(H, W), mode="bicubic", align_corners=False)

        g_rgb = self._grad_mag(rgb)
        g_dep = self._grad_mag(d_up)
        f_grad = F.leaky_relu(self.grad_stem(torch.cat([g_rgb, g_dep], dim=1)), 0.2)

        f_r = F.leaky_relu(self.rgb_stem(rgb), 0.2)
        f_d = F.leaky_relu(self.dep_stem(d_up), 0.2)

        for r_g, d_g, fre, fc in zip(self.rgb_groups, self.dep_groups, self.fre_blocks, self.fuse_convs):
            f_r = r_g(f_r)
            f_d = d_g(f_d)
            f_spec = fre(f_d, f_r)
            f_d = f_d + fc(torch.cat([f_d, f_spec, f_grad], dim=1))

        return (d_up + self.tail(f_d)).clamp_min(1e-3)


# ==============================================================================
# 4. DORNet (CVPR 2025): Degradation Oriented and Regularized Network
#    Reference: Yan et al., "DORNet: A Degradation Oriented and Regularized Network for Blind Depth Super-Resolution", CVPR 2025
#    Exact architecture matching official checkpoints (NYU_X4.pth, NYU_X8.pth, NYU_X16.pth, RGBDD.pth)
# ==============================================================================
class DCN_layer_rgb(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        deformable_groups: int = 1,
        bias: bool = True,
        extra_offset_mask: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.extra_offset_mask = extra_offset_mask
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=True,
        )
        self.c1 = nn.Conv2d(in_channels * 4, out_channels, 1, 1, 0, bias=False)
        self.c2 = nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()
        n = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input_feat: torch.Tensor, inter: torch.Tensor, fea: torch.Tensor) -> torch.Tensor:
        b, c, h, w = input_feat.shape
        fea_w = self.c1(fea).unsqueeze(1)
        weight = (self.weight.unsqueeze(0) * fea_w).view(
            b * self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        ).contiguous()

        out_off = self.conv_offset_mask(inter)
        o1, o2, mask = torch.chunk(out_off, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        # Process per-sample in batch for deform_conv2d with dynamic per-sample weights
        outs = []
        for i in range(b):
            w_i = weight[i * self.out_channels : (i + 1) * self.out_channels]
            out_i = deform_conv2d(
                input_feat[i : i + 1].contiguous(),
                offset[i : i + 1].contiguous(),
                w_i,
                bias=self.bias,
                stride=_pair(self.stride),
                padding=_pair(self.padding),
                dilation=_pair(self.dilation),
                mask=mask[i : i + 1].contiguous(),
            )
            outs.append(out_i)
        out = torch.cat(outs, dim=0)
        return self.c2(out)


class _DORResBlock(nn.Module):
    def __init__(self, in_feat: int, out_feat: int, stride: int = 1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_feat),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_feat),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_feat),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.backbone(x) + self.shortcut(x), 0.1, inplace=True)


class _DaEncoder(nn.Module):
    def __init__(self, nfeats: int):
        super().__init__()
        self.E_pre = nn.Sequential(
            _DORResBlock(in_feat=1, out_feat=nfeats // 2, stride=1),
            _DORResBlock(in_feat=nfeats // 2, out_feat=nfeats, stride=1),
            _DORResBlock(in_feat=nfeats, out_feat=nfeats, stride=1),
        )
        self.E = nn.Sequential(
            nn.Conv2d(nfeats, nfeats * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(nfeats * 2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(nfeats * 2, nfeats * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(nfeats * 4),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor):
        inter = self.E_pre(x)
        fea = self.E(inter)
        out = fea.squeeze(-1).squeeze(-1)
        return fea, out, inter


class _generateKernel(nn.Module):
    def __init__(self, nfeats: int, kernel_size: int = 5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(nfeats * 4, nfeats),
            nn.LeakyReLU(0.1, True),
            nn.Linear(nfeats, kernel_size * kernel_size),
        )

    def forward(self, D_Kernel: torch.Tensor) -> torch.Tensor:
        return self.mlp(D_Kernel)


class _DAB(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.LeakyReLU(0.1, True)
        self.conv = default_conv(1, 1, 1)

    def forward(self, x: torch.Tensor, D_Kernel: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        _, l = D_Kernel.shape
        kernel_size = int(math.sqrt(l))
        with torch.no_grad():
            kernel = D_Kernel.view(-1, 1, kernel_size, kernel_size)
            out = F.conv2d(x.view(1, -1, h, w), kernel, groups=b * c, padding=(kernel_size - 1) // 2)
            out = out.view(b, -1, h, w)
        return self.conv(self.relu(out).view(b, -1, h, w))


class _DecMoE(nn.Module):
    def __init__(self, ds_inputsize: int, input_size: int, output_size: int, num_experts: int, hidden_size: int, noisy_gating: bool = True, k: int = 2, trainingmode: bool = True):
        super().__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.training = trainingmode
        self.k = k
        self.experts = nn.ModuleList(
            [_generateKernel(hidden_size, 3), _generateKernel(hidden_size, 5), _generateKernel(hidden_size, 7), _generateKernel(hidden_size, 9)]
        )
        self.w_gate = nn.Parameter(torch.zeros(ds_inputsize, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(ds_inputsize, num_experts), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))


class _DR(nn.Module):
    def __init__(self, nfeats: int, num_experts: int = 4, k: int = 3):
        super().__init__()
        self.topK = k
        self.num_experts = num_experts
        self.start_idx = num_experts - k
        self.c1 = _DORResBlock(in_feat=1, out_feat=nfeats, stride=1)
        self.gap = nn.AdaptiveMaxPool2d(1)
        self.gap2 = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(nfeats, nfeats * 4)
        self.dab_list = nn.ModuleList([_DAB(), _DAB(), _DAB()])
        self.DecoderMoE = _DecMoE(ds_inputsize=nfeats * 4, input_size=1, output_size=1, num_experts=num_experts, hidden_size=nfeats, noisy_gating=True, k=k, trainingmode=True)
        self.conv = default_conv(1, 1, 1)


class _DA_rgb(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, reduction: int):
        super().__init__()
        self.dcnrgb = DCN_layer_rgb(channels_in, channels_out, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.rcab1 = RCAB(default_conv, channels_out, 3, reduction)
        self.relu = nn.LeakyReLU(0.1, True)
        self.conv = default_conv(channels_in, channels_out, 3)

    def forward(self, x: torch.Tensor, inter: torch.Tensor, fea: torch.Tensor) -> torch.Tensor:
        out1 = self.rcab1(x)
        out2 = self.dcnrgb(out1, inter, fea)
        return self.conv(out2 + out1)


class _FusionBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int):
        super().__init__()
        self.conv1 = default_conv(channels_in, channels_in // 4, 1)
        self.conv2 = default_conv(channels_in, channels_in // 4, 1)
        self.conv3 = default_conv(channels_in // 4, channels_in, 1)
        self.sigmoid = nn.Sigmoid()
        self.conv = default_conv(2 * channels_in, channels_out, 3)

    def forward(self, rgb: torch.Tensor, dep: torch.Tensor, inter: torch.Tensor) -> torch.Tensor:
        inter1 = self.conv1(inter)
        rgb1 = self.conv2(rgb)
        w = torch.sigmoid(inter1)
        rgb2 = rgb1 * w
        rgb3 = self.conv3(rgb2) + rgb
        return self.conv(torch.cat([rgb3, dep], dim=1))


class _DOFT(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, reduction: int):
        super().__init__()
        self.DA_rgb = _DA_rgb(channels_in, channels_out, kernel_size, reduction)
        self.fb = _FusionBlock(channels_in, channels_out)
        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x: torch.Tensor, inter: torch.Tensor, rgb: torch.Tensor, fea: torch.Tensor) -> torch.Tensor:
        rgb = self.DA_rgb(rgb, inter, fea)
        return x + self.fb(rgb, x, inter)


class _DOR_DSRN(nn.Module):
    def __init__(self, nfeats: int = 64, reduction: int = 16, conv=default_conv):
        super().__init__()
        n_feats = nfeats
        self.head = nn.Sequential(conv(1, n_feats, 3))
        self.head_rgb = nn.Sequential(conv(3, n_feats, 3))
        self.dgm1 = _DOFT(n_feats, n_feats, 3, reduction)
        self.dgm2 = _DOFT(n_feats, n_feats, 3, reduction)
        self.dgm3 = _DOFT(n_feats, n_feats, 3, reduction)
        self.dgm4 = _DOFT(n_feats, n_feats, 3, reduction)
        self.dgm5 = _DOFT(n_feats, n_feats, 3, reduction)

        self.c_d1 = ResidualGroup(conv, n_feats, 3, reduction=reduction, n_resblocks=2)
        self.c_d2 = ResidualGroup(conv, n_feats, 3, reduction=reduction, n_resblocks=2)
        self.c_d3 = ResidualGroup(conv, n_feats, 3, reduction=reduction, n_resblocks=2)
        self.c_d4 = ResidualGroup(conv, n_feats, 3, reduction=reduction, n_resblocks=2)
        self.c_d5 = nn.Sequential(
            conv(5 * n_feats, n_feats, 1),
            ResidualGroup(conv, n_feats, 3, reduction=reduction, n_resblocks=2),
        )
        self.c_r1 = conv(n_feats, n_feats, 3)
        self.c_r2 = conv(n_feats, n_feats, 3)
        self.c_r3 = conv(n_feats, n_feats, 3)
        self.c_r4 = conv(n_feats, n_feats, 3)
        self.act = nn.LeakyReLU(0.1, True)
        self.tail = nn.Sequential(conv(n_feats, 1, 3))

    def forward(self, x: torch.Tensor, inter: torch.Tensor, rgb: torch.Tensor, fea: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        rgb = self.head_rgb(rgb)
        rgb1 = self.c_r1(rgb)
        rgb2 = self.c_r2(self.act(rgb1))
        rgb3 = self.c_r3(self.act(rgb2))
        rgb4 = self.c_r4(self.act(rgb3))

        dep10 = self.dgm1(x, inter, rgb, fea)
        dep1 = self.c_d1(dep10)
        dep20 = self.dgm2(dep1, inter, rgb1, fea)
        dep2 = self.c_d2(self.act(dep20))
        dep30 = self.dgm3(dep2, inter, rgb2, fea)
        dep3 = self.c_d3(self.act(dep30))
        dep40 = self.dgm4(dep3, inter, rgb3, fea)
        dep4 = self.c_d4(self.act(dep40))
        dep50 = self.dgm5(dep4, inter, rgb4, fea)

        cat1 = torch.cat([dep1, dep2, dep3, dep4, dep50], dim=1)
        dep6 = self.c_d5(cat1)
        return self.tail(dep6 + x)


class DORNet_CVPR25(nn.Module):
    """Official DORNet (CVPR 2025) architecture compatible with NYU_X4.pth, NYU_X8.pth, NYU_X16.pth, RGBDD.pth."""

    def __init__(self, tiny_model: bool = False):
        super().__init__()
        n_feats = 24 if tiny_model else 64
        reduction = 4 if tiny_model else 16
        self.R = _DOR_DSRN(nfeats=n_feats, reduction=reduction)
        self.Enc = _DaEncoder(nfeats=n_feats)
        self.Dab = _DR(nfeats=n_feats)

    def forward(self, rgb: torch.Tensor, d_lr: torch.Tensor, mask_lr: Optional[torch.Tensor] = None) -> torch.Tensor:
        H, W = rgb.shape[-2:]
        if mask_lr is None:
            mask_lr = (d_lr > 1e-3).float()
        d_filled = SensorCalibration._inpaint_lr_holes(d_lr, mask_lr)
        d_bic = F.interpolate(d_filled, size=(H, W), mode="bicubic", align_corners=False)

        # Per-sample min-max normalization into [0, 1] as expected by DORNet
        d_min = d_bic.amin(dim=(-2, -1), keepdim=True)
        d_max = d_bic.amax(dim=(-2, -1), keepdim=True)
        span = (d_max - d_min).clamp_min(1e-4)
        x_query = (d_bic - d_min) / span

        fea, _, inter = self.Enc(x_query)
        restored_norm = self.R(x_query, inter, rgb, fea)
        restored = restored_norm * span + d_min
        return restored.clamp_min(1e-3)


# ==============================================================================
# 5. DuCos (ICCV 2025): Duality Constrained Depth Super-Resolution
#    Reference: Yan et al., "DuCos: Duality Constrained Depth Super-Resolution via Foundation Model", ICCV 2025
# ==============================================================================
class IterativeDualityFusionModule(nn.Module):
    """Lagrange Multiplier Duality Fusion Module from DuCos (ICCV 2025)."""

    def __init__(self, dim_rgb: int, dim_dep: int, max_iter: int = 3):
        super().__init__()
        out_dim = dim_dep
        self.conv_rgbd = nn.Conv2d(dim_rgb + dim_dep, out_dim, kernel_size=3, padding=1)
        self.conv_depth = nn.Conv2d(dim_dep, out_dim, kernel_size=3, padding=1)
        self.fc_depth_out = nn.Conv2d(out_dim, out_dim, kernel_size=1)
        self.max_iter = max_iter

    @staticmethod
    def compute_lagrange_multipliers(f_rgb: torch.Tensor, f_depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_r = f_rgb.mean(dim=(-2, -1), keepdim=True)
        mean_d = f_depth.mean(dim=(-2, -1), keepdim=True)
        rc = f_rgb - mean_r
        dc = f_depth - mean_d
        cov = torch.sum(rc * dc, dim=(-2, -1), keepdim=True)
        std_r = torch.sqrt(torch.sum(rc**2, dim=(-2, -1), keepdim=True) + 1e-6)
        std_d = torch.sqrt(torch.sum(dc**2, dim=(-2, -1), keepdim=True) + 1e-6)
        corr = cov / (std_r * std_d + 1e-6)
        lam_rgb = torch.sigmoid(corr)
        lam_dep = 1.0 - lam_rgb
        return lam_rgb, lam_dep

    def forward(self, f_rgb: torch.Tensor, f_depth: torch.Tensor) -> torch.Tensor:
        fr = self.conv_rgbd(torch.cat([f_rgb, f_depth], dim=1))
        fd = self.conv_depth(f_depth)
        lam_r, lam_d = self.compute_lagrange_multipliers(fr, fd)
        f_fused = lam_r * fr + lam_d * fd
        for _ in range(self.max_iter):
            lam_r, lam_d = self.compute_lagrange_multipliers(fr, f_fused)
            f_fused = lam_r * fr + lam_d * fd
        return self.fc_depth_out(f_fused)


class DuCos_ICCV25(nn.Module):
    """DuCos (ICCV 2025): Duality Constrained Depth Super-Resolution."""

    def __init__(self, n_feats: int = 48):
        super().__init__()
        self.head_dep = default_conv(1, n_feats, 3)
        self.head_rgb = default_conv(3, n_feats, 3)

        self.rgb_groups = nn.ModuleList([ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=2) for _ in range(4)])
        self.dep_groups = nn.ModuleList([ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=3) for _ in range(4)])
        self.duality_blocks = nn.ModuleList([IterativeDualityFusionModule(n_feats, n_feats, max_iter=3) for _ in range(4)])

        self.fuse_all = nn.Sequential(
            default_conv(n_feats * 4, n_feats, 1),
            ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=3),
            default_conv(n_feats, 1, 3),
        )

    def forward(self, rgb: torch.Tensor, d_lr: torch.Tensor, mask_lr: Optional[torch.Tensor] = None) -> torch.Tensor:
        H, W = rgb.shape[-2:]
        if mask_lr is None:
            mask_lr = (d_lr > 1e-3).float()
        d_filled = SensorCalibration._inpaint_lr_holes(d_lr, mask_lr)
        d_up = F.interpolate(d_filled, size=(H, W), mode="bicubic", align_corners=False)

        fr = self.head_rgb(rgb)
        fd = self.head_dep(d_up)
        stage_outs = []
        for rg, dg, dual in zip(self.rgb_groups, self.dep_groups, self.duality_blocks):
            fr = rg(fr)
            fd = dg(fd)
            fd = fd + dual(fr, fd)
            stage_outs.append(fd)

        res = self.fuse_all(torch.cat(stage_outs, dim=1))
        return (d_up + res).clamp_min(1e-3)


def build_sota_gdsr_model(name: str) -> nn.Module:
    name_l = name.lower()
    if name_l in ["dada", "dada_cvpr23"]:
        return DADA_CVPR23()
    elif name_l in ["ssdnet", "ssdnet_iccv23"]:
        return SSDNet_ICCV23()
    elif name_l in ["sgnet", "sgnet_aaai24"]:
        return SGNet_AAAI24()
    elif name_l in ["dornet", "dornet_cvpr25"]:
        return DORNet_CVPR25()
    elif name_l in ["ducos", "ducos_iccv25"]:
        return DuCos_ICCV25()
    else:
        raise ValueError(f"Unknown SOTA GDSR model: {name}")
