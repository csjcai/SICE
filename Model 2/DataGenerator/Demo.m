clear all; 
close all; 
clc;

InputSize = 320;          
stride    = 300;
count     = 1;
num       = 0;
folder  = 'Data/Data';
filepaths = dir(fullfile(folder, '*.JPG'));

folder1  = 'Data/Label';
filepaths1 = dir(fullfile(folder1, '*.JPG'));

dataL  = zeros(InputSize, InputSize, 4, 1,'single');
dataD  = zeros(InputSize, InputSize, 4, 1,'single');
label  = zeros(InputSize, InputSize, 3, 1,'single');
data   = zeros(InputSize, InputSize, 3, 1,'single');

for j = 1 :length(filepaths)
    disp(j)
    
    img_In = imread(fullfile(folder,filepaths(j).name)); 
    img_In = modcrop(img_In, 3);
    img_Out = imread(fullfile(folder1,filepaths1(j).name)); 
    img_Out = modcrop(img_Out, 3);
    
    img_In = imresize(img_In, 1/3, 'bicubic');
    img_Out = imresize(img_Out, 1/3, 'bicubic');
    
    [H, W, C] = size(img_In);
    
    if (H>320 && W>320)
    
        I = double(rgb2gray(img_In));
        I = I./max(I(:));
        lumin = im2single(wlsFilter(I, 2, 2));
        detail = im2single(I - lumin);
        
        img_In = im2single(img_In);
        img_Out = im2single(img_Out);
        DL = cat(3, img_In, lumin);
        DD = cat(3, img_In, detail);

        for x = 1 : stride : (H-InputSize+1)
            for y = 1 :stride : (W-InputSize+1)
                dataL(:, :, :, count)  = DL(x : x+InputSize-1, y : y+InputSize-1,:);
                dataD(:, :, :, count)  = DD(x : x+InputSize-1, y : y+InputSize-1,:);
                data(:, :, :, count)   = img_In(x : x+InputSize-1, y : y+InputSize-1,:);
                label(:, :, :, count)  = img_Out(x : x+InputSize-1, y : y+InputSize-1,:);
                count = count + 1;
            end
        end
        
        h5create('SICE.h5', '/dataL', size(dataL), 'Datatype', 'single')
        h5create('SICE.h5', '/dataD', size(dataD), 'Datatype', 'single')
        h5create('SICE.h5', '/label', size(label), 'Datatype', 'single')
        h5create('SICE.h5', '/data', size(data), 'Datatype', 'single')

        h5write('SICE.h5', '/dataL', single(dataL))
        h5write('SICE.h5', '/dataD', single(dataD))
        h5write('SICE.h5', '/label', single(label))
        h5write('SICE.h5', '/data', single(data))

    end
    
end
