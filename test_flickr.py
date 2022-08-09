import os
import cv2
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


def paddingSize(x):
    if x % 4 == 1:
        return 3
    elif x % 4 == 2:
        return 2
    elif x % 4 == 3:
        return 1
    else:
        return 0


def add_noise(src, alpha, sigma):
    if not alpha == 0:
        src = alpha * np.random.poisson(src / alpha).astype(float)
    noise = np.random.normal(0, sigma, src.shape)
    src = src + noise
    src = np.clip(src, 0, 1.0)
    return src


def fixSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
fixSeed(1234)


def main():
    # Noise Level
    alpha = 0.02
    sigma = 0.2

    # Load Image List
    img_path = '../../Datasets/Flickr1024/Test/'       # Root path of the Flickr1024 Dataset
    img_names = glob.glob(img_path + '*_L.png')
    img_names = sorted(img_names)
    img_num = len(img_names)

    save_dir = 'res/' + str('%.2f'%alpha) + '_' + str('%.2f'%sigma) + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load model
    model = SANet().cuda()
    model.load_state_dict(torch.load('models/SANet_flickr.pkl'))
    model.eval()

    # Metrics
    index = 0
    mpsnr = 0
    mssim = 0
    mlpips = 0
    lpfunc = lpips.LPIPS(net='vgg').cuda()
    
    # Start Denoising
    for img_name in img_names:
        strs = img_name.split('/')
        real_name = strs[-1]
        index += 1

        target_img = cv2.imread(img_path + real_name)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        target_img = np.array(target_img/255.0, dtype=float)

        guidance_img = cv2.imread(img_path + real_name.replace('_L', '_R'))
        guidance_img = cv2.cvtColor(guidance_img, cv2.COLOR_BGR2RGB)
        guidance_img = np.array(guidance_img/255.0, dtype=float)
        guidance_img = guidance_img[:, :, (1,2,0)]

        noisy_img = add_noise(target_img, alpha, sigma)
        noisy_img = np.clip(noisy_img, 0, 1.0)

        target_img = torch.from_numpy(np.ascontiguousarray(target_img)).permute(2, 0, 1).float().unsqueeze(0).cuda()
        guidance_img = torch.from_numpy(np.ascontiguousarray(guidance_img)).permute(2, 0, 1).float().unsqueeze(0)
        noisy_img = torch.from_numpy(np.ascontiguousarray(noisy_img)).permute(2, 0, 1).float().unsqueeze(0)

        h, w = target_img.shape[2], target_img.shape[3]
        h_psz = paddingSize(h)
        w_psz = paddingSize(w)
        padding = torch.nn.ReflectionPad2d((0, w_psz, 0, h_psz))
        noisy_img = padding(noisy_img)
        guidance_img = padding(guidance_img)

        with torch.no_grad():
            denoised_r = model(noisy_img[:,0,:,:,].unsqueeze(0).cuda(), guidance_img[:,0,:,:,].unsqueeze(0).cuda())
            denoised_g = model(noisy_img[:,1,:,:,].unsqueeze(0).cuda(), guidance_img[:,1,:,:,].unsqueeze(0).cuda())
            denoised_b = model(noisy_img[:,2,:,:,].unsqueeze(0).cuda(), guidance_img[:,2,:,:,].unsqueeze(0).cuda())

        denoised_img = torch.cat([denoised_r, denoised_g, denoised_b], dim=1)
        denoised_img = torch.clamp(denoised_img, 0, 1.0)
        denoised_img = denoised_img[:,:,:h,:w]

        lpips_value = lpfunc(denoised_img, target_img).item()
        mlpips += lpips_value

        denoised_img = denoised_img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        target_img = target_img.squeeze().permute(1, 2, 0).detach().cpu().numpy()

        psnr_value = psnr(denoised_img, target_img)
        ssim_value = ssim(denoised_img, target_img, multichannel=True)

        mpsnr += psnr_value
        mssim += ssim_value

        print(str(index).zfill(3), ':', psnr_value, ssim_value, lpips_value)

        im = Image.fromarray(np.uint8(denoised_img* 255))
        im.save(save_dir + real_name.replace('_L', '_res'))


    print('==================================================================')
    print(mpsnr / img_num)
    print(mssim / img_num)
    print(mlpips / img_num)


if __name__ == "__main__":
    main()