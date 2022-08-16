import os
import cv2
import math
import glob
import torch
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from SANet import SANet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = False


def paddingSize(x, d):
    return math.ceil(x / d) * d - x


def add_noise(src, alpha, sigma):
    if not alpha == 0:
        src = alpha * np.random.poisson(src / alpha).astype(float)
    noise = np.random.normal(0, sigma, src.shape)
    src = src + noise
    src = np.clip(src, 0, 1.0)
    return src


def main():
    # Set Noise Level
    alpha = 0
    sigma = 0.2

    # Load model
    model = SANet().cuda()
    model.StrAgg.load_state_dict(torch.load('models/structure_aggregation_flickr.pkl'))
    model.load_state_dict(torch.load('models/SANet_flickr.pkl'))
    model.eval()
    lpfunc = lpips.LPIPS(net='vgg').cuda()

    # Load Image - Flickr1024
    target_img = cv2.imread('examples/Flickr/01_target.png')
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    target_img = np.array(target_img/255.0, dtype=float)

    guidance_img = cv2.imread('examples/Flickr/01_guidance.png')
    guidance_img = cv2.cvtColor(guidance_img, cv2.COLOR_BGR2RGB)
    guidance_img = np.array(guidance_img/255.0, dtype=float)

    # Add Noise
    noisy_img = add_noise(target_img, alpha, sigma)
    noisy_img = np.clip(noisy_img, 0, 1.0)

    # Convert Images Into Tensors
    target_img = torch.from_numpy(np.ascontiguousarray(target_img)).permute(2, 0, 1).float().unsqueeze(0).cuda()
    guidance_img = torch.from_numpy(np.ascontiguousarray(guidance_img)).permute(2, 0, 1).float().unsqueeze(0).cuda()
    noisy_img = torch.from_numpy(np.ascontiguousarray(noisy_img)).permute(2, 0, 1).float().unsqueeze(0).cuda()

    # Conduct Image Padding
    h, w = target_img.shape[2], target_img.shape[3]
    h_psz = paddingSize(h, 4)
    w_psz = paddingSize(w, 4)
    padding = torch.nn.ReflectionPad2d((0, w_psz, 0, h_psz))
    noisy_img = padding(noisy_img)
    guidance_img = padding(guidance_img)

    # Start Denoising
    with torch.no_grad():
        denoised_r = model(noisy_img[:,0,:,:,].unsqueeze(0), guidance_img[:,0,:,:,].unsqueeze(0))
        denoised_g = model(noisy_img[:,1,:,:,].unsqueeze(0), guidance_img[:,1,:,:,].unsqueeze(0))
        denoised_b = model(noisy_img[:,2,:,:,].unsqueeze(0), guidance_img[:,2,:,:,].unsqueeze(0))
    denoised_img = torch.cat([denoised_r, denoised_g, denoised_b], dim=1)
    denoised_img = torch.clamp(denoised_img, 0, 1.0)
    denoised_img = denoised_img[:,:,:h,:w]
    noisy_img = noisy_img[:,:,:h,:w]

    # Compute PSNR, SSIM, & LPIPS
    lpips_value = lpfunc(denoised_img, target_img).item()

    denoised_img = denoised_img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    target_img = target_img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    noisy_img = noisy_img.squeeze().permute(1, 2, 0).detach().cpu().numpy()

    psnr_value = psnr(denoised_img, target_img)
    ssim_value = ssim(denoised_img, target_img, multichannel=True)

    print('==========================================================================')
    print('PSNR={}, SSIM={}, LPIPS={}'.format(psnr_value, ssim_value, lpips_value))
    print('==========================================================================')

    # Save Images
    im = Image.fromarray(np.uint8(denoised_img* 255))
    im.save('results/Denoising/01_flickr_res.png')

    im = Image.fromarray(np.uint8(noisy_img* 255))
    im.save('results/Denoising/01_flickr_noisy.png')


if __name__ == "__main__":
    main()