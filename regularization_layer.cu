#include <vector>
#include <string>
#include <algorithm>

#include "caffe/layers/regularization_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Table1(const int nthreads, const Dtype* const bottom_data, 
    Dtype* top_data, Dtype level1, Dtype value1, Dtype value2) {

     CUDA_KERNEL_LOOP(index, nthreads) {
     if (bottom_data[index] < 0){
        top_data[index] = 0;
     }
     else if ((bottom_data[index] >= 0) && (bottom_data[index] < level1)){
        top_data[index] = value1;
     }
     else if ((bottom_data[index] >= level1) && (bottom_data[index] <= 1)){
        top_data[index] = value2;
     }
     else{
        top_data[index] = 1;
     }
     }
}

template <typename Dtype>
__global__ void Table2(const int nthreads, const Dtype* const bottom_data, 
    Dtype* top_data, Dtype level1, Dtype level2, Dtype level3, Dtype value1, Dtype value2, Dtype value3, Dtype value4) {

     CUDA_KERNEL_LOOP(index, nthreads) {
     if (bottom_data[index] < 0){
        top_data[index] = 0;
     }
     else if ((bottom_data[index] >= 0) && (bottom_data[index] < level1)){
        top_data[index] = value1;
     }
     else if ((bottom_data[index] >= level1) && (bottom_data[index] < level2)){
        top_data[index] = value2;
     }
     else if ((bottom_data[index] >= level2) && (bottom_data[index] < level3)){
        top_data[index] = value3;
     }
     else if((bottom_data[index] >= level3) && (bottom_data[index] <= 1)){
        top_data[index] = value4;
     }
     else{
        top_data[index] = 1;
     }
     }
}

template <typename Dtype>
__global__ void Table3(const int nthreads, const Dtype* const bottom_data, 
    Dtype* top_data, Dtype level1, Dtype level2, Dtype level3, Dtype level4, Dtype level5, Dtype level6, Dtype level7, 
    Dtype value1, Dtype value2, Dtype value3, Dtype value4, Dtype value5, Dtype value6, Dtype value7, Dtype value8) {

     CUDA_KERNEL_LOOP(index, nthreads) {
     if (bottom_data[index] < 0){
        top_data[index] = 0;
     }
     else if ((bottom_data[index] >= 0) && (bottom_data[index] < level1)){
        top_data[index] = value1;
     }
     else if ((bottom_data[index] >= level1) && (bottom_data[index] < level2)){
        top_data[index] = value2;
     }
     else if ((bottom_data[index] >= level2) && (bottom_data[index] < level3)){
        top_data[index] = value3;
     }
     else if((bottom_data[index] >= level3) && (bottom_data[index] < level4)){
        top_data[index] = value4;
     }
     else if ((bottom_data[index] >= level4) && (bottom_data[index] < level5)){
        top_data[index] = value5;
     }
     else if ((bottom_data[index] >= level5) && (bottom_data[index] < level6)){
        top_data[index] = value6;
     }
     else if ((bottom_data[index] >= level6) && (bottom_data[index] < level7)){
        top_data[index] = value7;
     }
     else if((bottom_data[index] >= level7) && (bottom_data[index] <= 1)){
        top_data[index] = value8;
     }
     else{
        top_data[index] = 1;
     }
     }
}

template <typename Dtype>
void RegularizationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  Dtype* loss = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();

  switch (this->layer_param_.regularization_param().method()){
  case RegularizationParameter_ReMethod_Zero:
    {
       caffe_gpu_set(count, Dtype(regular_), tmp_.mutable_gpu_data());
    }
       break;
  case RegularizationParameter_ReMethod_One:
    {
        Table1<Dtype> <<< CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >>>
          (count, bottom_data, tmp_.mutable_gpu_data(), level1, value1, value2);
        CUDA_POST_KERNEL_CHECK;
    }
       break;
  case RegularizationParameter_ReMethod_Two:
    {
        Table2<Dtype> <<< CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >>>
          (count, bottom_data, tmp_.mutable_gpu_data(), level1, level2, level3, value1, value2, value3, value4);
        CUDA_POST_KERNEL_CHECK;
    }
       break;
  case RegularizationParameter_ReMethod_Three:
    {
        Table3<Dtype> <<< CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >>>
          (count, bottom_data, tmp_.mutable_gpu_data(), level1, level2, level3, level4, level5, level6, level7, value1, value2, value3, value4, value5, value6, value7, value8);
        CUDA_POST_KERNEL_CHECK;
    }
       break;
  default:
      LOG(FATAL) << "Unknow Method";
  }

  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      tmp_.gpu_data(),
      diff_.mutable_gpu_data());

  switch (this->layer_param_.regularization_param().norm()){
  case RegularizationParameter_Norm_L2:
    {
       Dtype dot;
       caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
       Dtype scale_L2 = 1.0 / (num) / Dtype(2);
       loss[0] = scale_L2 * dot;
    }
       break;
  case RegularizationParameter_Norm_L1:
    {
       Dtype asum;
       caffe_gpu_asum(count, diff_.gpu_data(), &asum);
       Dtype scale_L1 = 1.0 / (num) / Dtype(2);
       loss[0] = scale_L1 * asum;
    }
       break;
  default:
      LOG(FATAL) << "Unknow Norm";
  }
}

template <typename Dtype>
void RegularizationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  int count = bottom[0]->count();
  const Dtype sign = 1;
  const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[0]->num();

  switch (this->layer_param_.regularization_param().norm()){
  case RegularizationParameter_Norm_L2:
    { 
       caffe_gpu_set(count, Dtype(0), bottom[0]->mutable_gpu_diff());
       caffe_gpu_axpby(
          count, 
          alpha,  
          diff_.gpu_data(), 
          Dtype(0), 
          bottom[0]->mutable_gpu_diff());
    }
      break;
  case RegularizationParameter_Norm_L1:
    {  
       caffe_gpu_sign(count, diff_.gpu_data(), bottom[0]->mutable_gpu_diff()); 
       caffe_gpu_scale(
        count,
        alpha,
        diff_.gpu_data(), 
        bottom[0]->mutable_gpu_diff());
    }
       break;
  default:
      LOG(FATAL) << "Unknow Norm";
  }
}
INSTANTIATE_LAYER_GPU_FUNCS(RegularizationLayer);
}  // namespace caffe