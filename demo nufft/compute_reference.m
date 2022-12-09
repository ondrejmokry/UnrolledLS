% reference for single forward and backward NUFFT

clear
clc
close all

fileID = fopen('../datafold.txt','r');
datafold = fscanf(fileID,'%s');
fclose(fileID);

fileID = fopen('../packagefold.txt','r');
packagefold = fscanf(fileID,'%s');
fclose(fileID);

addpath(genpath([packagefold, '\ESPIRiT']));
rng(0)

% pre-set parameters
Nsamples = 128; % number of acquired samples per each radial line
X        = 512; % image width
Y        = 512; % image height
rads     = 64;  % number of radial lines

%% generate an image
margX = X/4;
margY = Y/4;
I = imread([datafold, '\panda.png']);
I = imresize(I, [Y-margY, X-margX]);
I = rgb2gray(I);
newI = zeros(Y,X);
newI(margY/2+1:end-margY/2,margX/2+1:end-margX/2) = I;
I = newI;
I = double(I)/255;

%% simulate the acquisition
golden = 2*pi /(1+sqrt(5));
angles = (0:rads-1)*golden;

% these are the acquired locations in the k-space
kx = ( (-Nsamples/2):(Nsamples/2-1) )' * cos(angles);
ky = ( (-Nsamples/2):(Nsamples/2-1) )' * sin(angles);
kx = kx/X;
ky = ky/Y;

% compute the density compensation
w = DoCalcDCF(kx(:), ky(:))' * X*Y;

% compute the nufft
FT = NUFFT(ky(:) + 1i*kx(:), w, [0 0], [Y X]);
nufftdata = FT*I;

%% simulate the reconstruction
recI = FT'*nufftdata;

%% save the data
save([datafold,'\nufft_reference.mat'],'I','nufftdata','Nsamples','X','Y','rads','w','recI')