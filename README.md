# Structure Aggregation for Cross-Spectral Stereo Image Guided Denoising (CVPR'2023)

## Abstract
To obtain clean images with salient structures from noisy observations, a growing trend in current denoising studies is to seek the help of additional guidance images with high signal-to-noise ratios, which are often acquired in different spectral bands such as near infrared. Although previous guided denoising methods basically require the input images to be well-aligned, a more common way to capture the paired noisy target and guidance images is to exploit a stereo camera system. However, current studies on cross-spectral stereo matching cannot fully guarantee the pixel-level registration accuracy, and rarely consider the case of noise contamination. In this work, for the first time, we propose a guided denoising framework for cross-spectral stereo images. Instead of aligning the input images via conventional stereo matching, we aggregate structures from the guidance image to estimate a clean structure map for the noisy target image, which is then used to regress the final denoising result with a spatially variant linear representation model. Based on this, we design a neural network, called as SANet, to complete the entire guided denoising process. Experimental results show that, our SANet can effectively transfer structures from an unaligned guidance image to the restoration result, and also outperforms state-of-the-art denoising algormithms on various stereo datasets. Besides, our structure aggregation strategy also shows its potential to handle other unaligned guided restoration tasks such as super-resolution and deblurring.

## Description
This model is built in PyTorch 1.6.0 and tested on Ubuntu 18.04 (Python 3.6.13, CUDA 10.0). At present, we provide our pre-trained models on the Flickr1024, the KITTI Stereo 2015, and the PittsStereo-RGBNIR Datasets for quick evaluation. Training codes will be made available in the future.


## Evaluation
We provide 3 demos to evaluate the example paired images from the Flickr1024 Dataset, the KITTI Stereo 2015 Dataset, the PittsStereo-RGBNIR Dataset, respectively. We also provide an additional demo to evaluate our own captured RGB-NIR stereo paired images with the model trained on the Flickr1024 training set.
```
# Evaluation on Flickr1024
python demo_test_denoising_flickr.py

# Evaluation on KITTI Stereo 2015
python demo_test_denoising_kitti.py

# Evaluation on PittsStereo-RGBNIR
python demo_test_denoising_pitts.py

# Evaluation on our captured RGB-NIR paired images
python demo_test_denoising_ours.py
```

## Acknowledgement
Our work is mainly evaluated on the Flickr1024, Kitti Stereo 2015, and the PittsStereo-RGBNIR Datasets. We thank the authors of the open datasets for their contributions.