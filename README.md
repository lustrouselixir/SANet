# Structure Aggregation for Cross-Spectral Stereo Image Guided Denoising
Zehua Sheng, Zhu Yu, Xiongwei Liu, Hui-Liang Shen, and Huaqi Zhang




## Description
This model is built in PyTorch 1.6.0 and tested on Ubuntu 18.04 (Python 3.6.13, CUDA 10.0). At present, we provide our pre-trained models on the Flickr1024, the KITTI Stereo 2015, and the PittsStereo-RGBNIR Datasets for quick evaluation. Training codes will be made available soon.


## Evaluation on Flickr1024
The Flickr1024 Dataset can be downloaded at https://yingqianwang.github.io/Flickr1024/.
Before evaluation, please modify the root path of the dataset and the noise level parameters in `test_flickr.py`. Then you can run
```
python test_flickr.py
```

## Evaluation on Single Data
We also provide an example paired images in the folder `example/` for evaluation. Just modify the image path and the noise level parameters, then run
```
python test_single_image.py
```

## Acknowledgement
Our work is mainly evaluated on the Flickr1024, Kitti Stereo 2015, and the PittsStereo-RGBNIR Datasets. We thank the authors of the open datasets for their contributions.