# Structure Aggregation for Cross-Spectral Stereo Image Guided Denoising
(Anonymous)

## Description
This model is built in PyTorch 1.6.0 and tested on Ubuntu 18.04 (Python 3.6.13, CUDA 10.0). At present, we provide our pre-trained models on the Flickr1024, the KITTI Stereo 2015, and the PittsStereo-RGBNIR Datasets for quick evaluation. Training codes will be made available soon.


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