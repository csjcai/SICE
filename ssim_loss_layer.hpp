#ifndef CAFFE_SSIM_LOSS_LAYER_HPP_
#define CAFFE_SSIM_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the SSIM loss @f$
 *          E = \frac{1}{N} \sum\limits_{p \in P} 1 - SSIM(p) 
 *        @f$ .
 *
 * @param bottom input Blob vector (length 2)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the predictions @f$ \hat{y} \in [-\infty, +\infty]@f$
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the targets @f$ y \in [-\infty, +\infty]@f$
 * @param top output Blob vector (length 1)
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *      the computed SSIM loss: @f$ E =
 *          \frac{1}{N} \sum\limits_{p \in P} 1 - SSIM(p) @f$ 
 */
template <typename Dtype>
class SSIMLossLayer : public LossLayer<Dtype> {
 public:
  explicit SSIMLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param){}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SSIMLoss"; }

 protected:
  /// @copydoc SSIMLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the SSIM error gradient w.r.t. the inputs.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      This Blob's diff will simply contain the loss_weight* @f$ \lambda @f$,
   *      as @f$ \lambda @f$ is the coefficient of this layer's output
   *      @f$\ell_i@f$ in the overall Net loss
   *      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
   *      @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
   *      (*Assuming that this top Blob is not used as a bottom (input) by any
   *      other layer of the Net.)
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the predictions @f$\hat{y}@f$; Backward fills their diff with
   *      gradients @f$
   *        \frac{\partial E}{\partial \hat{y}} =
   *            \frac{1}{N} \sum\limits_{n=1}^N sign(\hat{y}_n - y_n)
   *      @f$ if propagate_down[0]
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the targets @f$y@f$; Backward fills their diff with gradients
   *      @f$ \frac{\partial E}{\partial y} =
   *          \frac{1}{N} \sum\limits_{n=1}^N sign(y_n - \hat{y}_n)
   *      @f$ if propagate_down[1]
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  void GaussConvolveHelper(const Blob<Dtype>& in, Blob<Dtype>& out);
  void CudaGaussConvolveHelper(const Blob<Dtype>& in, Blob<Dtype>& out);

  Blob<Dtype> ux_;
  Blob<Dtype> uy_;
  Blob<Dtype> sx2_;
  Blob<Dtype> sy2_;
  Blob<Dtype> sxy_;
  Blob<Dtype> lp_;
  Blob<Dtype> cs_;
  //Does storing the denominators work out better than recomputing it
  //later on ?
  //Blob<Dtype> lp_den_;
  //Blob<Dtype> cs_den_;
  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  int channels_;
  int height_, width_;
  int pooled_height_, pooled_width_;
  Blob<double> gauss_kernel_;
  double c1_, c2_;
};

}  // namespace caffe

#endif  // CAFFE_SSIM_LOSS_LAYER_HPP_