# Learning a Deep Single Image Contrast Enhancer from Multi-Exposure Images

## Abstract
Due to the poor lighting condition and limited dynamic range of digital imaging devices, the recorded images are often under-/over-exposed and with low contrast. Most of previous single image contrast enhancement (SICE) methods adjust the tone curve to correct the contrast of an input image. Those methods, however, often fail in revealing image details because of the limited information in a single image. On the other hand, the SICE task can be better accomplished if we can learn extra information from appropriately collected training data. In this work, we propose to use the convolutional neural network (CNN) to train a SICE enhancer. One key issue is how to construct a training dataset of low-contrast and high-contrast image pairs for end-to-end CNN learning. To this end, we build a large-scale multi-exposure image dataset, which contains 589 elaborately selected high-resolution multi-exposure sequences with 4,413 images. Thirteen representative multi-exposure image fusion and stack-based high dynamic range imaging algorithms are employed to generate the contrast enhanced images for each sequence, and subjective experiments are conducted to screen the best quality one as the reference image of each scene. With the constructed dataset, a CNN can be easily trained as the SICE enhancer to improve the contrast of an under-/over-exposure image. Experimental results demonstrate the advantages of our method over existing SICE methods with a significant margin.

## Dataset
Please refer to: 
1) Google Drive:
2) BaiduYun:

## Requirements and Dependencies
* Cuda-8.0 & cuDNN -V5.1
* Deep Learning Toolboxs (Caffe, Tensorflow, Pytorch, MatConvnet)

## Caffe 
### New Layers With CPU and GPU Implementations
#### L1 Loss Layer

#### SSIM Loss Layer

##### Usage
```
layer {
  name: "mylosslayer"
  type: "SSIMLoss"
  bottom: "result"
  bottom: "ground_truth"
  top: "loss_vale"
  loss_weight: 1             # <- set whatever you fancy
  ssim_loss_param{
    kernel_size: 8           # <- The kernel size is linked to the gaussian variance (circular). The kernel encloses +/1 3*sigma 
    stride: 8                # <- Equal strides in both dimensions
    c1: 0.0001               # <- Let these be
    c2: 0.001                # <- Let these be
  }
}
```


