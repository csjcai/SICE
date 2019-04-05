#include <vector>
#include <string>
#include <algorithm>

#include "caffe/layers/regularization_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RegularizationLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  diff_.ReshapeLike(*bottom[0]);
  tmp_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void RegularizationLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  RegularizationParameter rep = this->layer_param_.regularization_param();
  
  regular_ = rep.regular();

  level1 = rep.level1();
  level2 = rep.level2();
  level3 = rep.level3();
  level4 = rep.level4();
  level5 = rep.level5();
  level6 = rep.level6();
  level7 = rep.level7();

  value1 = rep.value1();
  value2 = rep.value2();
  value3 = rep.value3();
  value4 = rep.value4();
  value5 = rep.value5();
  value6 = rep.value6();
  value7 = rep.value7();
  value8 = rep.value8();

}

template <typename Dtype>
void RegularizationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  int count = bottom[0]->count();
  int num = bottom[0]->num();
  Dtype* loss = top[0]->mutable_cpu_data();
  const Dtype* const bottom_data = bottom[0]->cpu_data();
  Dtype* tmp = tmp_.mutable_cpu_data();

  switch (this->layer_param_.regularization_param().method()){
  case RegularizationParameter_ReMethod_Zero:
    {
       caffe_set(count, Dtype(regular_), tmp_.mutable_cpu_data());
    }
       break;
  case RegularizationParameter_ReMethod_One:
    {
       for (int i = 0; i < bottom[0]->count(); i++){
         if (bottom_data[i] < 0){        
           tmp[i] = 0;
         }
         else if ((bottom_data[i] >= 0) && (bottom_data[i] < level1)){     
           tmp[i] = value1;
         }
         else if ((bottom_data[i] >= level1) && (bottom_data[i] <= 1)){        
           tmp[i] = value2;
         }
         else{            
           tmp[i] = 1;
         }
      }
    }
       break;
  case RegularizationParameter_ReMethod_Two:
    {
       for (int i = 0; i < bottom[0]->count(); i++){
         if (bottom_data[i] < 0){        
           tmp[i] = 0;
         }
         else if ((bottom_data[i] >= 0) && (bottom_data[i] < level1)){     
           tmp[i] = value1;
         }
         else if ((bottom_data[i] >= level1) && (bottom_data[i] < level2)){        
           tmp[i] = value2;
         }
         else if ((bottom_data[i] >= level2) && (bottom_data[i] < level3)){        
           tmp[i] = value3;
         }
         else if((bottom_data[i] >= level3) && (bottom_data[i] <= 1)){     
           tmp[i] = value4;
         }
         else{            
           tmp[i] = 1;
         }
      }
    }
       break;
  case RegularizationParameter_ReMethod_Three:
    {
       for (int i = 0; i < bottom[0]->count(); i++){
         if (bottom_data[i] < 0){        
           tmp[i] = 0;
         }
         else if ((bottom_data[i] >= 0) && (bottom_data[i] < level1)){     
           tmp[i] = value1;
         }
         else if ((bottom_data[i] >= level1) && (bottom_data[i] < level2)){        
           tmp[i] = value2;
         }
         else if ((bottom_data[i] >= level2) && (bottom_data[i] < level3)){        
           tmp[i] = value3;
         }
         else if((bottom_data[i] >= level3) && (bottom_data[i] < level4)){     
           tmp[i] = value4;
         }
         else if ((bottom_data[i] >= level4) && (bottom_data[i] < level5)){     
           tmp[i] = value5;
         }
         else if ((bottom_data[i] >= level5) && (bottom_data[i] < level6)){        
           tmp[i] = value6;
         }
         else if ((bottom_data[i] >= level6) && (bottom_data[i] < level7)){        
           tmp[i] = value7;
         }
         else if((bottom_data[i] >= level7) && (bottom_data[i] <= 1)){     
           tmp[i] = value8;
         }
         else{            
           tmp[i] = 1;
         }
      }
    }
       break;
  default:
      LOG(FATAL) << "Unknow Method";
  }

  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      tmp_.cpu_data(),
      diff_.mutable_cpu_data());

  switch (this->layer_param_.regularization_param().norm()){
  case RegularizationParameter_Norm_L2:
    {
       Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
       Dtype scale_L2 = 1.0 / (num) / Dtype(2);
       loss[0] = scale_L2 * dot;
    }
       break;
  case RegularizationParameter_Norm_L1:
    {
       Dtype asum = caffe_cpu_asum(count, diff_.cpu_data());
       Dtype scale_L1 = 1.0 / (num) / Dtype(2);
       loss[0] = scale_L1 * asum;
    }
       break;
  default:
      LOG(FATAL) << "Unknow Norm";
  }
}

template <typename Dtype>
void RegularizationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  int count = bottom[0]->count();
  const Dtype sign = 1;
  const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[0]->num();

  switch (this->layer_param_.regularization_param().norm()){
  case RegularizationParameter_Norm_L2:
    { 
       caffe_set(count, Dtype(0), bottom[0]->mutable_cpu_diff());
       caffe_cpu_axpby(
          count, 
          alpha,  
          diff_.cpu_data(), 
          Dtype(0), 
          bottom[0]->mutable_cpu_diff());
    }
      break;
  case RegularizationParameter_Norm_L1:
    {  
       caffe_cpu_sign(count, bottom[0]->cpu_data(), bottom[0]->mutable_cpu_diff()); 
       caffe_cpu_scale(
        count,
        alpha,
        diff_.cpu_data(), 
        bottom[0]->mutable_cpu_diff());
    }
       break;
  default:
      LOG(FATAL) << "Unknow Norm";
  }
}

#ifdef CPU_ONLY
STUB_GPU(RegularizationLayer);
#endif

INSTANTIATE_CLASS(RegularizationLayer);
REGISTER_LAYER_CLASS(Regularization);

}  // namespace caffe
