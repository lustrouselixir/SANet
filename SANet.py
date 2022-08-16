import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import initDCTKernel, initIDCTKernel


def tensor_roll(src, max_d):
    src = src.view(src.shape[0], src.shape[1], 1, src.shape[2], src.shape[3])
    guidance_rolled = src
    for i in range(1, max_d):
        guidance_slice = torch.roll(src, i, 4)
        guidance_rolled = torch.cat([guidance_rolled, guidance_slice], 2)
    return guidance_rolled


def image_roll(src, max_d):
    guidance_rolled = src
    for i in range(1, max_d):
        guidance_slice = torch.roll(src, i, 3)
        guidance_rolled = torch.cat([guidance_rolled, guidance_slice], 1)
    return guidance_rolled


def conv_3x3(in_channel, out_channel, stride=1, bias=False, padding=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=padding, bias=bias)


def conv3d_3x3(in_channel, out_channel, depth_channel, stride=1, bias=False, padding=1):
    return nn.Conv3d(in_channel, out_channel, kernel_size=(depth_channel, 3, 3), stride=stride, padding=padding, bias=bias)


def conv_3x3_dilated(in_channel, out_channel, stride=1, bias=False, padding=2):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=padding, bias=bias, dilation=2)


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


########### Structure Aggregation Module ###########
class Structure_Aggregation(nn.Module):
    def __init__(self):
        super(Structure_Aggregation, self).__init__()

        # Parameters
        C1 = 48
        C2 = 96
        C3 = 24
        self.max_disparity = 128
        D = int(self.max_disparity / 4)

        # Encoder Layers for the Noisy Target Image
        self.conv_1t = nn.Sequential(conv_3x3(1, C1), nn.GELU(), LayerNorm2d(C1))
        self.conv_2t = nn.Sequential(conv_3x3(C1, C2), nn.GELU(), LayerNorm2d(C2))
        self.conv_3t = nn.Sequential(conv_3x3(C2, C2), nn.GELU(), LayerNorm2d(C2))
        self.conv_4t = nn.Sequential(conv_3x3(C2, C2), nn.GELU(), LayerNorm2d(C2))
        self.conv_5t = nn.Sequential(conv_3x3(C2, C2), nn.GELU(), LayerNorm2d(C2))
        self.conv_6t = nn.Sequential(conv_3x3(C2, C3), nn.GELU(), LayerNorm2d(C3))

        # Encoder Layers for the Guidance Image
        self.conv_1g = nn.Sequential(conv_3x3(1, C1), nn.GELU(), LayerNorm2d(C1))
        self.conv_2g = nn.Sequential(conv_3x3(C1, C2), nn.GELU(), LayerNorm2d(C2))
        self.conv_3g = nn.Sequential(conv_3x3(C2, C2), nn.GELU(), LayerNorm2d(C2))
        self.conv_4g = nn.Sequential(conv_3x3(C2, C2), nn.GELU(), LayerNorm2d(C2))
        self.conv_5g = nn.Sequential(conv_3x3(C2, C2), nn.GELU(), LayerNorm2d(C2))
        self.conv_6g = nn.Sequential(conv_3x3(C2, C3), nn.GELU(), LayerNorm2d(C3))

        # Decoder Layers for the Perceptual Weight W^P
        self.conv_1w = nn.Sequential(conv_3x3(D, C2), nn.GELU(), LayerNorm2d(C2))
        self.conv_2w = nn.Sequential(conv_3x3(C2, C2), nn.GELU(), LayerNorm2d(C2))
        self.conv_3w = nn.Sequential(conv_3x3(C2, C2), nn.GELU(), LayerNorm2d(C2))
        self.conv_4w = nn.Sequential(conv_3x3(C2, C2), nn.GELU(), LayerNorm2d(C2))
        self.conv_5w = nn.Sequential(conv_3x3(C2, self.max_disparity), nn.GELU(), LayerNorm2d(self.max_disparity))

        # Other Layers
        self.max_pooling = nn.MaxPool2d(2)
        self.up_sampling = torch.nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, noisy, guidance):
        guidance_rolled = image_roll(guidance, self.max_disparity)

        # Encoder
        feat_t = self.max_pooling(noisy)
        feat_t = self.conv_1t(feat_t)
        feat_t = self.max_pooling(feat_t)
        feat_t = self.conv_2t(feat_t)
        feat_t = self.conv_3t(feat_t)
        feat_t = self.conv_4t(feat_t)
        feat_t = self.conv_5t(feat_t)
        feat_t = self.conv_6t(feat_t)

        B, C, H, W = feat_t.shape[0], feat_t.shape[1], feat_t.shape[2], feat_t.shape[3]
        D = int(self.max_disparity / 4)

        feat_t = feat_t.contiguous().view(B, C, H*W).permute(1, 0, 2)
        feat_t = feat_t.contiguous().view(C, 1, B*H*W).permute(2, 1, 0)

        feat_g = self.max_pooling(guidance)
        feat_g = self.conv_1g(feat_g)
        feat_g = self.max_pooling(feat_g)
        feat_g = self.conv_2g(feat_g)
        feat_g = self.conv_3g(feat_g)
        feat_g = self.conv_4g(feat_g)
        feat_g = self.conv_5g(feat_g)
        feat_g = self.conv_6g(feat_g)
        feat_g = tensor_roll(feat_g, D)

        feat_g = feat_g.contiguous().view(B, C, D, H*W).permute(1, 2, 0, 3)
        feat_g = feat_g.contiguous().view(C, D, B*H*W).permute(2, 0, 1)

        # Cross-Correlation
        feat_tg = torch.bmm(feat_t, feat_g)
        feat_tg = feat_tg.contiguous().view(B*H*W, D).permute(1, 0)
        feat_tg = feat_tg.contiguous().view(D, B, H*W)
        feat_tg = feat_tg.contiguous().view(D, B, H, W).permute(1, 0, 2, 3)

        # Decoder
        feat_tg = self.up_sampling(feat_tg)
        feat_tg = self.conv_1w(feat_tg)
        feat_tg = self.conv_2w(feat_tg)
        feat_tg = self.conv_3w(feat_tg)
        feat_tg = self.up_sampling(feat_tg)
        feat_tg = self.conv_4w(feat_tg)
        feat_tg = self.conv_5w(feat_tg)
        w_g = F.softmax(feat_tg, dim=1)

        structure_map = torch.sum(w_g*guidance_rolled, dim=1, keepdim=True)

        return structure_map


########### Guided Denoising Module - Noise Estimation ###########
class Noise_Estimation(nn.Module):
    def __init__(self):
        super(Noise_Estimation, self).__init__()
        C = 64

        self.conv1_1 = nn.Sequential(conv_3x3(1, C), nn.GELU(), LayerNorm2d(C))
        self.conv1_2 = nn.Sequential(conv_3x3(C, C), nn.GELU(), LayerNorm2d(C))
        self.conv1_3 = nn.Sequential(conv_3x3(C, C), nn.GELU(), LayerNorm2d(C))
        self.conv1_4 = nn.Sequential(conv_3x3(C, C), nn.GELU(), LayerNorm2d(C))
        self.conv1_5 = nn.Sequential(conv_3x3(C, C), nn.GELU(), LayerNorm2d(C))
        self.conv1_6 = nn.Sequential(conv_3x3(C, C), nn.GELU(), LayerNorm2d(C))
        self.conv1_7 = nn.Sequential(conv_3x3(C, C), nn.GELU(), LayerNorm2d(C))
        self.conv1_8 = nn.Sequential(conv_3x3(C, C), nn.GELU(), LayerNorm2d(C))
        self.conv1_9 = nn.Sequential(conv_3x3(C, C), nn.GELU(), LayerNorm2d(C))
        self.conv1_10 = nn.Sequential(conv_3x3(C, C), nn.GELU(), LayerNorm2d(C))
        self.conv1_11 = nn.Sequential(conv_3x3(C, C), nn.GELU(), LayerNorm2d(C))
        self.conv1_12 = nn.Sequential(conv_3x3(C, C), nn.GELU(), LayerNorm2d(C))
        self.conv1_13 = nn.Sequential(conv_3x3(C, C), nn.GELU(), LayerNorm2d(C))
        self.conv1_14 = nn.Sequential(conv_3x3(C, C), nn.GELU(), LayerNorm2d(C))
        self.conv1_15 = nn.Sequential(conv_3x3(C, C), nn.GELU(), LayerNorm2d(C))
        self.conv1_16 = nn.Conv2d(in_channels=C, out_channels=1, kernel_size=3, padding=1)

    def forward(self, src):
        feat = self.conv1_1(src)
        feat = self.conv1_2(feat)
        feat = self.conv1_3(feat)
        feat = self.conv1_4(feat)
        feat = self.conv1_5(feat)
        feat = self.conv1_6(feat)
        feat = self.conv1_7(feat)
        feat = self.conv1_8(feat)
        feat = self.conv1_9(feat)
        feat = self.conv1_10(feat)
        feat = self.conv1_11(feat)
        feat = self.conv1_12(feat)
        feat = self.conv1_13(feat)
        feat = self.conv1_14(feat)
        feat = self.conv1_15(feat)
        out = self.conv1_16(feat)
        out = src - out
        return out


########### SANet - Guided Denoising - Linear Representation ###########
class SANet(nn.Module):
    def __init__(self):
        super(SANet, self).__init__()

        # Parameters
        C1 = 96
        C2 = 128
        C3 = 96
        self.kernel_size = 9
        self.channelNum = self.kernel_size*self.kernel_size

        # Generate DCT & IDCT kernels
        in_kernel = initDCTKernel(self.kernel_size)
        out_kernel = initIDCTKernel(self.kernel_size)
        in_kernel = torch.Tensor(in_kernel)
        out_kernel = torch.Tensor(out_kernel)
        self.in_kernel = nn.Parameter(in_kernel)
        self.out_kernel = nn.Parameter(out_kernel)
        self.in_kernel.requires_grad = False
        self.out_kernel.requires_grad = False

        # Encoder Layers for the Noisy Target Image
        self.conv_1t = nn.Sequential(conv_3x3(self.channelNum, C1), nn.GELU(), LayerNorm2d(C1))
        self.conv_2t = nn.Sequential(conv_3x3(C1, C2), nn.GELU(), LayerNorm2d(C2))
        self.conv_3t = nn.Sequential(conv_3x3(C2, C2), nn.GELU(), LayerNorm2d(C2))
        self.conv_4t = nn.Sequential(conv_3x3(C2, C2), nn.GELU(), LayerNorm2d(C2))
        self.conv_5t = nn.Sequential(conv_3x3(C2, C2), nn.GELU(), LayerNorm2d(C2))
        self.conv_6t = nn.Sequential(conv_3x3(C2, C1), nn.GELU(), LayerNorm2d(C1))

        # Encoder Layers for the Structure Map
        self.conv_1g = nn.Sequential(conv_3x3(self.channelNum, C1), nn.GELU(), LayerNorm2d(C1))
        self.conv_2g = nn.Sequential(conv_3x3(C1, C2), nn.GELU(), LayerNorm2d(C2))
        self.conv_3g = nn.Sequential(conv_3x3(C2, C2), nn.GELU(), LayerNorm2d(C2))
        self.conv_4g = nn.Sequential(conv_3x3(C2, C2), nn.GELU(), LayerNorm2d(C2))
        self.conv_5g = nn.Sequential(conv_3x3(C2, C2), nn.GELU(), LayerNorm2d(C2))
        self.conv_6g = nn.Sequential(conv_3x3(C2, C1), nn.GELU(), LayerNorm2d(C1))

        # Encoder Layers for the Estimated Noise
        self.conv_1n = nn.Sequential(conv_3x3(self.channelNum, C1), nn.GELU(), LayerNorm2d(C1))
        self.conv_2n = nn.Sequential(conv_3x3(C1, C2), nn.GELU(), LayerNorm2d(C2))
        self.conv_3n = nn.Sequential(conv_3x3(C2, C2), nn.GELU(), LayerNorm2d(C2))
        self.conv_4n = nn.Sequential(conv_3x3(C2, C2), nn.GELU(), LayerNorm2d(C2))
        self.conv_5n = nn.Sequential(conv_3x3(C2, C2), nn.GELU(), LayerNorm2d(C2))
        self.conv_6n = nn.Sequential(conv_3x3(C2, C1), nn.GELU(), LayerNorm2d(C1))

        # Decoder Layers for W^S
        self.conv_1dt = nn.Sequential(conv_3x3(C1*3, C3), nn.GELU(), LayerNorm2d(C3))
        self.conv_2dt = nn.Sequential(conv_3x3(C3, C3), nn.GELU(), LayerNorm2d(C3))
        self.conv_3dt = nn.Sequential(conv_3x3(C3, C3), nn.GELU(), LayerNorm2d(C3)) 
        self.conv_4dt = nn.Sequential(conv_3x3(C3, self.channelNum), nn.GELU(), LayerNorm2d(self.channelNum))
        self.conv_5dt = nn.Sequential(conv_3x3(self.channelNum, self.channelNum), nn.Tanh())

        # Decoder Layers for W^G
        self.conv_1dg = nn.Sequential(conv_3x3(C1*3, C3), nn.GELU(), LayerNorm2d(C3))
        self.conv_2dg = nn.Sequential(conv_3x3(C3, C3), nn.GELU(), LayerNorm2d(C3))
        self.conv_3dg = nn.Sequential(conv_3x3(C3, C3), nn.GELU(), LayerNorm2d(C3)) 
        self.conv_4dg = nn.Sequential(conv_3x3(C3, self.channelNum), nn.GELU(), LayerNorm2d(self.channelNum))
        self.conv_5dg = nn.Sequential(conv_3x3(self.channelNum, self.channelNum), nn.Tanh())

        # Decoder Layers for W^N
        self.conv_1dn = nn.Sequential(conv_3x3(C1*3, C3), nn.GELU(), LayerNorm2d(C3))
        self.conv_2dn = nn.Sequential(conv_3x3(C3, C3), nn.GELU(), LayerNorm2d(C3))
        self.conv_3dn = nn.Sequential(conv_3x3(C3, C3), nn.GELU(), LayerNorm2d(C3)) 
        self.conv_4dn = nn.Sequential(conv_3x3(C3, self.channelNum), nn.GELU(), LayerNorm2d(self.channelNum))
        self.conv_5dn = nn.Sequential(conv_3x3(self.channelNum, self.channelNum), nn.Tanh())

        # Load Models
        self.NoiseEst = Noise_Estimation()
        self.StrAgg = Structure_Aggregation()

        # Other Layers
        self.max_pooling = nn.MaxPool2d(2)
        self.up_sampling = torch.nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, noisy, guidance):
        pre_denoised = self.NoiseEst(noisy)
        guidance_warped = self.StrAgg(noisy, guidance)
        noise = noisy - pre_denoised

        # Encoder
        feat_t0 = F.conv2d(input=noisy, weight=self.in_kernel, padding=self.kernel_size-1)
        feat_t = self.max_pooling(feat_t0)
        feat_t = self.conv_1t(feat_t)
        feat_t = self.max_pooling(feat_t)
        feat_t = self.conv_2t(feat_t)
        feat_t = self.conv_3t(feat_t)
        feat_t = self.conv_4t(feat_t)
        feat_t = self.conv_5t(feat_t)
        feat_t = self.conv_6t(feat_t)

        feat_g0 = F.conv2d(input=guidance_warped, weight=self.in_kernel, padding=self.kernel_size-1)
        feat_g = self.max_pooling(feat_g0)  
        feat_g = self.conv_1g(feat_g)
        feat_g = self.max_pooling(feat_g)
        feat_g = self.conv_2g(feat_g)
        feat_g = self.conv_3g(feat_g)
        feat_g = self.conv_4g(feat_g)
        feat_g = self.conv_5g(feat_g)
        feat_g = self.conv_6g(feat_g)

        feat_n0 = F.conv2d(input=noise, weight=self.in_kernel, padding=self.kernel_size-1)
        feat_n = self.max_pooling(feat_n0)
        feat_n = self.conv_1t(feat_n)
        feat_n = self.max_pooling(feat_n)
        feat_n = self.conv_2t(feat_n)
        feat_n = self.conv_3t(feat_n)
        feat_n = self.conv_4t(feat_n)
        feat_n = self.conv_5t(feat_n)
        feat_n = self.conv_6t(feat_n)

        feat_tgn = torch.cat([feat_t ,feat_g, feat_n], dim=1)
        feat_tgn = self.up_sampling(feat_tgn)


        # Decoder
        weight_t = self.conv_1dt(feat_tgn)
        weight_t = self.conv_2dt(weight_t)
        weight_t = self.conv_3dt(weight_t)
        weight_t = self.conv_4dt(weight_t)
        weight_t = self.up_sampling(weight_t)
        weight_t = self.conv_5dt(weight_t)

        weight_g = self.conv_1dg(feat_tgn)
        weight_g = self.conv_2dg(weight_g)
        weight_g = self.conv_3dg(weight_g)
        weight_g = self.conv_4dg(weight_g)
        weight_g = self.up_sampling(weight_g)
        weight_g = self.conv_5dg(weight_g)

        weight_n = self.conv_1dn(feat_tgn)
        weight_n = self.conv_2dn(weight_n)
        weight_n = self.conv_3dn(weight_n)
        weight_n = self.conv_4dn(weight_n)
        weight_n = self.up_sampling(weight_n)
        weight_n = self.conv_5dn(weight_n)

        out_freq = feat_t0 * weight_t + feat_g0 * weight_g + feat_n0 * weight_n
        out = F.conv2d(input=out_freq, weight=self.out_kernel, padding=0)

        return out
