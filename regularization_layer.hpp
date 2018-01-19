#ifndef CAFFE_REGULARIZATION_LAYER_HPP_
#define CAFFE_REGULARIZATION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
template <typename Dtype>
class RegularizationLayer : public LossLayer<Dtype> {
	public:
		explicit RegularizationLayer(const LayerParameter& param)
            : LossLayer<Dtype>(param), diff_() {}
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      		const vector<Blob<Dtype>*>& top);
		virtual inline const char* type() const { return "Regularization"; }
		virtual inline bool AllowForceBackward(const int bottom_index) const {
    		return true;
  		}

	 protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		    const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	        const vector<Blob<Dtype>*>& top);

		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	Blob<Dtype> tmp_;
	Blob<Dtype> diff_;
	
	Dtype regular_;
	
	Dtype level1;
	Dtype level2;
	Dtype level3;
	Dtype level4;
	Dtype level5;
	Dtype level6;
	Dtype level7;

	Dtype value1;
	Dtype value2;
	Dtype value3;
	Dtype value4;
	Dtype value5;
	Dtype value6;
	Dtype value7;
	Dtype value8;
	};

}  // namespace caffe

#endif  // CAFFE_REGULARIZATION_LAYER_HPP_
