import torch
import torchvision
import numpy as np
import math
import torch.nn.functional as F


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


class TVLoss(torch.nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


def initDCTKernel(N):
    kernel = np.zeros((N, N, N*N))
    cnum = 0
    for i in range(N):
        for j in range(N):
            ivec = np.linspace(0.5 * math.pi / N * i, (N - 0.5) * math.pi / N * i, num=N)
            ivec = np.cos(ivec)
            jvec = np.linspace(0.5 * math.pi / N * j, (N - 0.5) * math.pi / N * j, num=N)
            jvec = np.cos(jvec)
            slice = np.outer(ivec, jvec)

            if i==0 and j==0:
                slice = slice / N
            elif i*j==0:
                slice = slice * np.sqrt(2) / N
            else:
                slice = slice * 2.0 / N

            kernel[:,:,cnum] = slice
            cnum = cnum + 1
    kernel = kernel[np.newaxis, :]
    kernel = np.transpose(kernel, (3,0,1,2))
    return kernel


def initIDCTKernel(N):
    kernel = np.zeros((N, N, N*N))
    for i_ in range(N):
        i = N - i_ - 1
        for j_ in range(N):
            j = N - j_ - 1
            ivec = np.linspace(0, (i+0.5)*math.pi/N * (N-1), num=N)
            ivec = np.cos(ivec)
            jvec = np.linspace(0, (j+0.5)*math.pi/N * (N-1), num=N)
            jvec = np.cos(jvec)
            slice = np.outer(ivec, jvec)

            ic = np.sqrt(2.0 / N) * np.ones(N)
            ic[0] = np.sqrt(1.0 / N)
            jc = np.sqrt(2.0 / N) * np.ones(N)
            jc[0] = np.sqrt(1.0 / N)
            cmatrix = np.outer(ic, jc)

            slice = slice * cmatrix
            slice = slice.reshape((1, N*N))
            slice = slice[np.newaxis, :]
            kernel[i_, j_, :] = slice / (N * N)
    kernel = kernel[np.newaxis, :]
    kernel = np.transpose(kernel, (0,3,1,2))
    return kernel


class DCT(torch.nn.Module):
    def __init__(self,ksz):
        super(DCT, self).__init__()
        self.kernel_size = ksz
        in_kernel = initDCTKernel(self.kernel_size)
        in_kernel = torch.Tensor(in_kernel)
        self.in_kernel = torch.nn.Parameter(in_kernel)
        self.in_kernel.requires_grad = False

    def forward(self, x):
        out = F.conv2d(input=x, weight=self.in_kernel, padding=self.kernel_size-1)
        return out