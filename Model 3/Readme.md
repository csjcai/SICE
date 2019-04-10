## Run Demo_Test.m (Compile the Caffe and the MatCaffe first)

## This is the new method for the image enhancement.

You should first complie your caffe with the PixelConv_layer




### Add the followeing line to caffe.proto
message PixelConvParameter {

  optional bool is_pad = 1 [default = true];
  
  optional bool is_bpk = 2 [default = true];
  
  optional bool is_bpd = 3 [default = true];
  
}

![image](https://github.com/csjcai/SICE/blob/master/Model%203/model3.bmp)
