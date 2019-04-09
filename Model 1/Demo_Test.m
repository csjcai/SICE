clear;
clc;
close all;
warning off;

addpath(genpath('./.'));
addpath(genpath('/home/jerry/caffe/')) ;

caffe.set_mode_gpu();
caffe.set_device(0);

folder  = 'Model/';
filepaths = dir(fullfile(folder, '*.caffemodel'));

weights = fullfile(folder,filepaths.name);
model = 'residual.prototxt';
net = caffe.Net(model, weights,'test');


img_In = imread('Test/79.JPG'); 
img_In = im2single(img_In);
img_In = imresize(img_In, 1/3, 'bicubic');
img_In = modcrop(img_In, 4);

net.blobs('data').reshape([size(img_In,1) size(img_In,2), 3, 1]);
net.reshape();

res = net.forward({img_In});

result = res{1};
imshow([img_In result])
