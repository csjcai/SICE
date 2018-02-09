# Learning a Deep Single Image Contrast Enhancer from Multi-Exposure Images

## Abstract
Due to the poor lighting condition and limited dynamic range of digital imaging devices, the recorded images are often under-/over-exposed and with low contrast. Most of previous single image contrast enhancement (SICE) methods adjust the tone curve to correct the contrast of an input image. Those methods, however, often fail in revealing image details because of the limited information in a single image. On the other hand, the SICE task can be better accomplished if we can learn extra information from appropriately collected training data. In this work, we propose to use the convolutional neural network (CNN) to train a SICE enhancer. One key issue is how to construct a training dataset of low-contrast and high-contrast image pairs for end-to-end CNN learning. To this end, we build a large-scale multi-exposure image dataset, which contains 589 elaborately selected high-resolution multi-exposure sequences with 4,413 images. Thirteen representative multi-exposure image fusion and stack-based high dynamic range imaging algorithms are employed to generate the contrast enhanced images for each sequence, and subjective experiments are conducted to screen the best quality one as the reference image of each scene. With the constructed dataset, a CNN can be easily trained as the SICE enhancer to improve the contrast of an under-/over-exposure image. Experimental results demonstrate the advantages of our method over existing SICE methods with a significant margin.

## Dataset
Please refer to: 
* Google Drive: [Part1: 360 Image Sequences](https://goo.gl/gTGfLk)

or

* BaiduYun: [Part1: 360 Image Sequences](https://pan.baidu.com/s/1kXotehL)

## Requirements and Dependencies
- [Cuda](https://developer.nvidia.com/cuda-toolkit-archive)-8.0 & [cuDNN](https://developer.nvidia.com/cudnn) v-5.1
- Deep Learning Toolboxs ([Caffe](https://github.com/BVLC/caffe), [Tensorflow](https://github.com/tensorflow/tensorflow), [Pytorch](https://github.com/pytorch/pytorch), [MatConvNet](http://www.vlfeat.org/matconvnet/))

## Caffe 
### New Layers With CPU and GPU Implementations
#### L1 Loss Layer
https://github.com/csjcai/SICE/blob/master/L1_loss_layer.hpp

https://github.com/csjcai/SICE/blob/master/L1_loss_layer.cpp

https://github.com/csjcai/SICE/blob/master/L1_loss_layer.cu

#### Regularization Layer
https://github.com/csjcai/SICE/blob/master/regularization_layer.hpp

https://github.com/csjcai/SICE/blob/master/regularization_layer.cpp

https://github.com/csjcai/SICE/blob/master/regularization_layer.cu

#### SSIM Loss Layer
https://github.com/csjcai/SICE/blob/master/ssim_loss_layer.hpp

https://github.com/csjcai/SICE/blob/master/ssim_loss_layer.cpp

https://github.com/csjcai/SICE/blob/master/ssim_loss_layer.cu

#### caffe.proto (Parameters for SSIM and Regularization Layer)
https://github.com/csjcai/SICE/blob/master/caffe.proto

##### Usage
```
layer {
  name: "SSIMLossLayer"
  type: "SSIMLoss"
  bottom: "output"
  bottom: "label"
  top: "SSIMLoss"
  ssim_loss_param{
    kernel_size: 8       
    stride: 8                
    c1: 0.0001              
    c2: 0.001                
  }
}
```


## Relevant Networks
- [Fast Image Processing with Fully-Convolutional Networks (ICCV 2017)](http://www.cqf.io/papers/Fast_Image_Processing_ICCV2017.pdf)          (https://github.com/CQFIO/FastImageProcessing)

- [Deep Bilateral Learning for Real-Time Image Enhancements (Siggraph 2017)](https://groups.csail.mit.edu/graphics/hdrnet/data/hdrnet.pdf)
     (https://github.com/mgharbi/hdrnet)



## Citation

```
@article{Cai2018deep,
title={Learning a Deep Single Image Contrast Enhancer from Multi-Exposure Images}, 
author={Cai, Jianrui and Gu, Shuhang and Zhang, Lei},
journal={IEEE Transactions on Image Processing},
volume={27}, 
number={4}, 
pages={2049-2062}, 
year={2018}, 
publisher={IEEE}
```
