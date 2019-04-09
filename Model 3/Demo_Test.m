clear;
clc;
close all;
warning off;

addpath(genpath('./.'));
addpath(genpath('/home/../caffe/')) ;

caffe.set_mode_gpu();
caffe.set_device(0);

folder  = 'Model/';
filepaths = dir(fullfile(folder, '*.caffemodel'));

weights = fullfile(folder,filepaths.name);
model = 'Perpixel.prototxt';
net = caffe.Net(model, weights,'test');


img_In = imread('Test/79.JPG'); 
img_In = imresize(img_In, 3/5, 'bicubic');
img_In = modcrop(img_In, 4);


I = double(rgb2gray(img_In));
I = I./max(I(:));
lumin = im2single(wlsFilter(I, 2, 2));
detail = im2single(I - lumin);

img_In = im2single(img_In);
DL = cat(3, img_In, lumin);
DD = cat(3, img_In, detail);

[H, W, C] = size(img_In);


if (H>1200 && W>1200)
    dataL = DL(H/2-600:H/2+599, W/2-600:W/2+599, :);
    dataD = DD(H/2-600:H/2+599, W/2-600:W/2+599, :);
    data = img_In(H/2-600:H/2+599, W/2-600:W/2+599, :);
end


net.blobs('data').reshape([size(data,1) size(data,2), 3, 1]);
net.reshape();

net.blobs('dataL').reshape([size(data,1) size(data,2), 4, 1]);
net.reshape();

net.blobs('dataD').reshape([size(data,1) size(data,2), 4, 1]);
net.reshape();

res = net.forward({data, dataL, dataD});

result = res{1};
imshow([data result])
