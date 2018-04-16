#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/ssim_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void SSIMLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  SSIMLossParameter ssim_param = this->layer_param_.ssim_loss_param();
  CHECK(ssim_param.has_kernel_size()) 
      << "Patch Size and Gauss SIGMA derive from kernel_size.";
  CHECK(ssim_param.has_stride())
      << "Stride in both dimensions is the same, and derives from stride.";
  CHECK(ssim_param.has_c1())
      << "Constant C1.";
  CHECK(ssim_param.has_c2())
      << "Constant C2.";
  c1_ = ssim_param.c1();
  c2_ = ssim_param.c2();
  CHECK_GT(c1_, 0) << "Abs(c1) cannot be zero.";
  CHECK_GT(c2_, 0) << "Abs(c2) cannot be zero.";
  kernel_h_ = kernel_w_ = ssim_param.kernel_size();
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  stride_h_ = stride_w_ = ssim_param.stride();
  CHECK_GT(stride_h_, 0) << "Stride cannot be zero.";

  gauss_kernel_.Reshape(1,1,kernel_h_,kernel_w_);
  double* gaussian = gauss_kernel_.mutable_cpu_data();
  double sigma = (kernel_w_+kernel_h_)/Dtype(12);
  double gauss_sum = 0;

  for (int h = 0; h < kernel_h_; ++h) {
      for (int w = 0; w < kernel_w_; ++w) {
    gaussian[h * kernel_w_ + w] = 
      exp(-(pow((h - kernel_h_/2.0),2) + pow((w - kernel_w_/2.0),2)) / (2.0* sigma * sigma))
          / (2 * 3.14159 * sigma * sigma);
     gauss_sum += gaussian[h* kernel_w_ + w];
      }
  }
  caffe_scal(gauss_kernel_.count(), 1.0/gauss_sum, gaussian);
}

template <typename Dtype>
void SSIMLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) //why was it count(1) here?
      << "Inputs must have the same dimension.";
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input0 must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  CHECK_EQ(4, bottom[1]->num_axes()) << "Input1 must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ - kernel_h_) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ - kernel_w_) / stride_w_)) + 1;
  ux_.Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  uy_.ReshapeLike(ux_);
  sx2_.ReshapeLike(ux_);
  sy2_.ReshapeLike(ux_);
  sxy_.ReshapeLike(ux_);
  lp_.ReshapeLike(ux_);
  cs_.ReshapeLike(ux_);
}


template <typename Dtype>
void SSIMLossLayer<Dtype>::GaussConvolveHelper(const Blob<Dtype>& in, Blob<Dtype>& out){
    int N = in.num();
    const Dtype* in_data = in.cpu_data();
    Dtype* out_data = out.mutable_cpu_data();
    caffe_set(out.count(), Dtype(0), out_data);
    const double* gaussian = gauss_kernel_.cpu_data();
    for (int n = 0; n < N; ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ ;
            int wstart = pw * stride_w_ ;
            int hend = min(hstart + kernel_h_, height_ );
            int wend = min(wstart + kernel_w_, width_ );
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                out_data[ph * pooled_width_ + pw] +=
                    gaussian[(h-hstart)*kernel_w_+(w-wstart)]*in_data[h * width_ + w];
              }
            }
    }
        }
        in_data += in.offset(0, 1);
        out_data += out.offset(0, 1);
      }
    }
}

template <typename Dtype>
void SSIMLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {  
  int count = bottom[0]->count();
  
  GaussConvolveHelper(*bottom[0],ux_);
  GaussConvolveHelper(*bottom[1],uy_);

  Blob<Dtype> tempContainer1, tempContainer2;
  tempContainer1.ReshapeLike(*bottom[0]);
  caffe_sqr(count, bottom[0]->cpu_data(), tempContainer1.mutable_cpu_data()); 
  GaussConvolveHelper(tempContainer1,sx2_);
  caffe_sqr(count, bottom[1]->cpu_data(), tempContainer1.mutable_cpu_data()); 
  GaussConvolveHelper(tempContainer1,sy2_);
  caffe_mul(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), tempContainer1.mutable_cpu_data()); 
  GaussConvolveHelper(tempContainer1,sxy_);

  tempContainer1.ReshapeLike(ux_);
  tempContainer2.ReshapeLike(uy_);
  count = tempContainer1.count();

  //Compute ux^2 and uy^2 and collect ux^2+uy^2 for later use
  caffe_sqr(count, ux_.cpu_data(), tempContainer1.mutable_cpu_data());
  caffe_sub(count, sx2_.cpu_data(), tempContainer1.cpu_data(), sx2_.mutable_cpu_data());
  caffe_sqr(count, uy_.cpu_data(), tempContainer2.mutable_cpu_data());
  caffe_sub(count, sy2_.cpu_data(), tempContainer2.cpu_data(), sy2_.mutable_cpu_data());
  caffe_add(count, tempContainer1.cpu_data(), tempContainer2.cpu_data(), tempContainer2.mutable_cpu_data());

  caffe_mul(count, ux_.cpu_data(), uy_.cpu_data(), tempContainer1.mutable_cpu_data());
  caffe_sub(count, sxy_.cpu_data(), tempContainer1.cpu_data(), sxy_.mutable_cpu_data());
  
  const Dtype C1 = c1_;
  caffe_cpu_scale(count, Dtype(2), tempContainer1.cpu_data(), tempContainer1.mutable_cpu_data());
  caffe_add_scalar(count, C1, tempContainer1.mutable_cpu_data());
  caffe_add_scalar(count, C1, tempContainer2.mutable_cpu_data());
  caffe_div(count, tempContainer1.cpu_data(), tempContainer2.cpu_data(), lp_.mutable_cpu_data());

  const Dtype C2 = c2_;
  caffe_add(count, sx2_.cpu_data(), sy2_.cpu_data(), tempContainer2.mutable_cpu_data()); 
  caffe_add_scalar(count, C2, tempContainer2.mutable_cpu_data());
  caffe_cpu_scale(count, Dtype(2), sxy_.cpu_data(), tempContainer1.mutable_cpu_data());
//  caffe_cpu_axpby(count, Dtype(2), sxy_.cpu_data(), Dtype(0), tempContainer1.mutable_cpu_data());
  caffe_add_scalar(count, C2, tempContainer1.mutable_cpu_data());
  caffe_div(count, tempContainer1.cpu_data(), tempContainer2.cpu_data(), cs_.mutable_cpu_data());
  
  Dtype ssim = caffe_cpu_dot(count, lp_.cpu_data(),cs_.cpu_data()) ;
  //Dtype ssim = caffe_cpu_dot(count, lp_.cpu_data(),cs_.cpu_data()) / ux_.count();
  Dtype loss = (Dtype(count)-ssim)/ bottom[0]->num();
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void SSIMLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* x = bottom[0]->cpu_data();
  const Dtype* y = bottom[1]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* ux = ux_.cpu_data();
  const Dtype* uy = uy_.cpu_data();
  const Dtype* lp = lp_.cpu_data();
  const Dtype* cs = cs_.cpu_data();
  const Dtype* sx2 = sx2_.cpu_data();
  const Dtype* sy2 = sy2_.cpu_data();
  const Dtype* sxy = sxy_.cpu_data();
  const double* gaussian = gauss_kernel_.cpu_data();

  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  const Dtype alpha = -top[0]->cpu_diff()[0] / bottom[0]->num();
  //const Dtype alpha = -top[0]->cpu_diff()[0] / ux_.count();

  // The main loop
  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
         for (int pw = 0; pw < pooled_width_; ++pw) {
      int hstart = ph * stride_h_ ;
      int wstart = pw * stride_w_ ;
      int hend = min(hstart + kernel_h_, height_);
      int wend = min(wstart + kernel_w_, width_ );
      int p = ph * pooled_width_ + pw;
      Dtype deriv1 = Dtype(2) * cs[p] * (uy[p]-ux[p]*lp[p]) / (ux[p]*ux[p]+uy[p]*uy[p]+Dtype(c1_));
      Dtype deriv2 = Dtype(2) * lp[p]  / (sx2[p]+sy2[p]+Dtype(c2_));
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
    int q = h * width_ + w;
    bottom_diff[q] += 
                    gaussian[(h-hstart)*kernel_w_+(w-wstart)]* ( deriv1 + ( deriv2 * ((y[q] - uy[p]) - cs[p]*(x[q]-ux[p]))));
        }
      }
    }
  }
  // offset
  x+= bottom[0]->offset(0,1);
  y+= bottom[1]->offset(0,1);
  bottom_diff += bottom[0]->offset(0, 1);
  ux+= ux_.offset(0,1);
  uy+= ux_.offset(0,1);
  sx2+= ux_.offset(0,1);
  sy2+= ux_.offset(0,1);
  sxy+= ux_.offset(0,1);
  lp+= ux_.offset(0,1);
  cs+= ux_.offset(0,1);
    }
  }
  caffe_cpu_scale(
          bottom[0]->count(),              // count
          alpha,                              // alpha
          bottom[0]->cpu_diff(),              // x
          bottom_diff);  // y
}

#ifdef CPU_ONLY
STUB_GPU(SSIMLossLayer);
#endif

INSTANTIATE_CLASS(SSIMLossLayer);
REGISTER_LAYER_CLASS(SSIMLoss);

}  // namespace caffe
