#include <vector>

#include "caffe/layers/L1_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void L1LossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  Dtype asum;
  caffe_gpu_asum(count, diff_.gpu_data(), &asum);
  Dtype loss = asum / bottom[0]->num() ;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void L1LossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      //Using the volume of the blob to normalize so that the loss is
      //independent of the image size or number of channels
      //const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->count();
      caffe_gpu_sign(bottom[i]->count(), diff_.gpu_data(),
      		     bottom[i]->mutable_gpu_diff());
      caffe_gpu_scale(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          bottom[i]->gpu_diff(),                   // x
          bottom[i]->mutable_gpu_diff());  // y
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(L1LossLayer);

}  // namespace caffe
