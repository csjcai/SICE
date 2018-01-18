#include <vector>

#include "caffe/layers/ssim_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
__global__ void CudaGaussConvolution(const int nthreads,
    const Dtype* const in_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, 
    const double* const gauss_kernel, Dtype* const out_data ) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h;
    int wstart = pw * stride_w;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    Dtype aveval = 0;
    const Dtype* const in_slice =
        in_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += gauss_kernel[(h-hstart)*kernel_w+(w-wstart)] * in_slice[h * width + w];
      }
    }
    out_data[index] = aveval;
  }
}


template <typename Dtype>
void SSIMLossLayer<Dtype>::CudaGaussConvolveHelper(const Blob<Dtype>& in,
    Blob<Dtype>& out) { 
  //Parallelized on the # of outputs to be produced
  CudaGaussConvolution<Dtype><<<CAFFE_GET_BLOCKS(out.count()), CAFFE_CUDA_NUM_THREADS>>>(
        out.count(), in.gpu_data(), in.num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, gauss_kernel_.gpu_data(),
  out.mutable_gpu_data() 
  );
}

//These be scars from the debug process. They tell a tale!
//Don't remove them.
/*
template <typename Dtype>
bool isNaN(Dtype *ptr, size_t n)
{
    for(int k = 0; k < n; ++k)
        if (ptr[k] != ptr[k])
      return true;
    return false;
}
*/

template <typename Dtype>
void SSIMLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {  
  int count = bottom[0]->count();
 // LOG(INFO) << "x " << isNaN(bottom[0]->cpu_data(), bottom[0]->count()) << std::endl;
  CudaGaussConvolveHelper(*bottom[0],ux_);
  CudaGaussConvolveHelper(*bottom[1],uy_);

  Blob<Dtype> tempContainer1, tempContainer2;
  tempContainer1.ReshapeLike(*bottom[0]);
  caffe_gpu_powx(count, bottom[0]->gpu_data(), Dtype(2), tempContainer1.mutable_gpu_data()); 
  CudaGaussConvolveHelper(tempContainer1,sx2_);
  caffe_gpu_powx(count, bottom[1]->gpu_data(), Dtype(2), tempContainer1.mutable_gpu_data()); 
  CudaGaussConvolveHelper(tempContainer1,sy2_);
  caffe_gpu_mul(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), tempContainer1.mutable_gpu_data()); 
  CudaGaussConvolveHelper(tempContainer1,sxy_);

  tempContainer1.ReshapeLike(ux_);
  tempContainer2.ReshapeLike(uy_);
  count = tempContainer1.count();

  //Compute ux^2 and uy^2 and collect ux^2+uy^2 for later use
  caffe_gpu_powx(count, ux_.gpu_data(), Dtype(2), tempContainer1.mutable_gpu_data());
  caffe_gpu_sub(count, sx2_.gpu_data(), tempContainer1.gpu_data(), sx2_.mutable_gpu_data());
  caffe_gpu_powx(count, uy_.gpu_data(), Dtype(2), tempContainer2.mutable_gpu_data());
  caffe_gpu_sub(count, sy2_.gpu_data(), tempContainer2.gpu_data(), sy2_.mutable_gpu_data());
  caffe_gpu_add(count, tempContainer1.gpu_data(), tempContainer2.gpu_data(), tempContainer2.mutable_gpu_data());

  caffe_gpu_mul(count, ux_.gpu_data(), uy_.gpu_data(), tempContainer1.mutable_gpu_data());
  caffe_gpu_sub(count, sxy_.gpu_data(), tempContainer1.gpu_data(), sxy_.mutable_gpu_data());
  
  //More scars from the debug process. Scars maketh a battle hardened warrior.
  //LOG(INFO) << "ux_ " << isNaN(ux_.cpu_data(), ux_.count()) << std::endl;
  //LOG(INFO) << "uy_ " << isNaN(uy_.cpu_data(), uy_.count()) << std::endl;
  //LOG(INFO) << "sx2_ " << isNaN(sx2_.cpu_data(), sx2_.count()) << std::endl;
  //LOG(INFO) << "sy2_ " << isNaN(sy2_.cpu_data(), sy2_.count()) << std::endl;
  //LOG(INFO) << "sxy_ " << isNaN(sxy_.cpu_data(), sxy_.count()) << std::endl;
  const Dtype C1 = c1_;
  caffe_gpu_scale(count, Dtype(2), tempContainer1.gpu_data(), tempContainer1.mutable_gpu_data());
  caffe_gpu_add_scalar(count, C1, tempContainer1.mutable_gpu_data());
  caffe_gpu_add_scalar(count, C1, tempContainer2.mutable_gpu_data());
  caffe_gpu_div(count, tempContainer1.gpu_data(), tempContainer2.gpu_data(), lp_.mutable_gpu_data());

  const Dtype C2 = c2_;
  caffe_gpu_add(count, sx2_.gpu_data(), sy2_.gpu_data(), tempContainer2.mutable_gpu_data()); 
  caffe_gpu_add_scalar(count, C2, tempContainer2.mutable_gpu_data());
  caffe_gpu_scale(count, Dtype(2), sxy_.gpu_data(), tempContainer1.mutable_gpu_data());
  caffe_gpu_add_scalar(count, C2, tempContainer1.mutable_gpu_data());
  caffe_gpu_div(count, tempContainer1.gpu_data(), tempContainer2.gpu_data(), cs_.mutable_gpu_data());
  
  //LOG(INFO) << "cs_ " << isNaN(cs_.cpu_data(), cs_.count()) << std::endl;
  //LOG(INFO) << "lp_ " << isNaN(lp_.cpu_data(), lp_.count()) << std::endl;

  Dtype ssim;
  caffe_gpu_dot(count, lp_.gpu_data(),cs_.gpu_data(), &ssim);
  //ssim/= bottom[0]->num();
  //ssim/= ux_.count();
  Dtype loss = Dtype(count)-ssim;
  top[0]->mutable_cpu_data()[0] = loss/bottom[0]->num();
  //LOG(INFO) << "loss " <<isNaN(top[0]->cpu_data(),1) << std::endl; 
}

template <typename Dtype>
__global__ void SSIMBackward(const int nthreads, 
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const Dtype* const x,const Dtype* const y,
    const Dtype* const ux,const Dtype* const uy,
    const Dtype* const sx2, const Dtype* const sy2, const Dtype* const sxy,
    const Dtype* const lp, const Dtype* const cs, Dtype c1, Dtype c2,  // <- These motherduckers were int for some reason! Gaaaaah!!!!
    const double* const gaussian,  Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width ;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    int p_inc = (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
  int p = p_inc + ph * pooled_width + pw;
        int hstart = ph * stride_h ;
        int wstart = pw * stride_w ;
        int hend = min(hstart + kernel_h, height );
        int wend = min(wstart + kernel_w, width);
  Dtype deriv1 = Dtype(2) * cs[p] * (uy[p]-ux[p]*lp[p]) / (ux[p]*ux[p]+uy[p]*uy[p]+c1);
  Dtype deriv2 = Dtype(2) * lp[p]  / (sx2[p]+sy2[p]+c2);
        gradient += gaussian[(h-hstart)*kernel_w+(w-wstart)]* ( deriv1 + ( deriv2 * ((y[index] - uy[p]) - cs[p]*(x[index]-ux[p]))));
      }
    }
    bottom_diff[index] = gradient;
  }
}
template <typename Dtype>
void SSIMLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* x = bottom[0]->gpu_data();
//  LOG(INFO) << "back x " << isNaN(bottom[0]->cpu_data(), bottom[0]->count()) << std::endl;
  const Dtype* y = bottom[1]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* ux = ux_.gpu_data();
  const Dtype* uy = uy_.gpu_data();
  const Dtype* lp = lp_.gpu_data();
  const Dtype* cs = cs_.gpu_data();
  const Dtype* sx2 = sx2_.gpu_data();
  const Dtype* sy2 = sy2_.gpu_data();
  const Dtype* sxy = sxy_.gpu_data();
  const double* gaussian = gauss_kernel_.gpu_data();

  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);
  const Dtype alpha = -top[0]->cpu_diff()[0] / bottom[0]->num();
  //const Dtype alpha = -top[0]->gpu_diff()[0] / ux_.count();

  //LOG(INFO) << "alpha " << isNaN(&alpha, 1) << std::endl;
  //Parallelized on bottom, ie, q
  SSIMBackward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        bottom[0]->count(), bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, x, y, ux, uy, sx2, sy2, sxy, lp, cs,
  c1_, c2_, gaussian,
  bottom_diff 
  );
  //LOG(INFO) << "bottom_diff avg asum " << caffe_cpu_asum(bottom[0]->count(), bottom[0]->cpu_diff())/bottom[0]->count() << std::endl;
  caffe_gpu_scale(
          bottom[0]->count(),              // count
          alpha,                              // alpha
          bottom[0]->gpu_diff(),              // x
          bottom_diff);  // y
  // Look up Rana Sanga. The dude had a lot of scars. Was a battle hardened ruler.
  //LOG(INFO) << "bottom_diff " << isNaN(bottom[0]->cpu_diff(), bottom[0]->count()) << std::endl;
  //LOG(INFO) << "bottom_diff avg asum " << caffe_cpu_asum(bottom[0]->count(), bottom[0]->cpu_diff())/bottom[0]->count() << std::endl;

}

INSTANTIATE_LAYER_GPU_FUNCS(SSIMLossLayer);

}  // namespace caffe