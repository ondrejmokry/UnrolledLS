% the script illustrates what happens with the spectrum when the image is
% subsampled
%
% Date: 18/02/2021
% By Ondrej Mokry
% Brno University of Technology
% Contact: ondrej.mokry@mensa.cz

clear
clc
close all

% add the package
fileID = fopen('../packagefold.txt','r');
packagefold = fscanf(fileID,'%s');
fclose(fileID);
addpath(genpath([packagefold, '/ESPIRiT']));

fileID = fopen('../datafold.txt','r');
datafold = fscanf(fileID,'%s');
fclose(fileID);

addpath('reconstruction');
rng(0)

% pre-set parameters
X = 1024;       % image width
Y = 1024;       % image height
factor = 8;     % factor of width and height reduction in the reconstruction
n = 2;          % index of the radial line
Nsamples = 128; % number of acquired samples on the radial line

%% generate an image
I = imread([datafold, '/panda.png']);
I = imresize(I, [Y-256 X-256]);
I = rgb2gray(I);
subI = zeros(Y,X);
subI(129:end-128,129:end-128) = I;
I = subI;
I = double(I);
I = I/max(I(:)) + 0.25i*rand(Y,X);

%% simulate the acquisition along a single radial
% these are the acquired locations in the k-space
golden = 2*pi /(1+sqrt(5));
angle  = n*golden;
kx     = ( (-Nsamples/2):(Nsamples/2-1) )' * cos(angle);
ky     = ( (-Nsamples/2):(Nsamples/2-1) )' * sin(angle);
kx     = kx(:)/X;
ky     = ky(:)/Y;

% compute the linear interpolation of the spectrum
[XX,YY]   = meshgrid(1:X,1:Y);
XX        = XX - X/2 - 1;
YY        = YY - Y/2 - 1;
fulldata  = fftshift(fft2(fftshift(I)));
partdata  = interp2(XX,YY,fulldata,X*kx,Y*ky,'linear');
    
%% subsample the image and compute its spectrum
% subsample
subX = X/factor;
subY = Y/factor;
subI = imresize(I, 1/factor);

% compute the k-space locations
subkx = ( (-Nsamples/2):(Nsamples/2-1) )' * cos(angle);
subky = ( (-Nsamples/2):(Nsamples/2-1) )' * sin(angle);
subkx = subkx(:)/subX;
subky = subky(:)/subY;

% compute the linear interpolation of the spectrum
[subXX,subYY] = meshgrid(1:subX,1:subY);
subXX         = subXX - subX/2 - 1;
subYY         = subYY - subY/2 - 1;
subfulldata   = fftshift(fft2(fftshift(subI)));
subpartdata   = interp2(subXX,subYY,subfulldata,subX*subkx,subY*subky,'linear');

%% plot
figure

% full image
subplot(2,3,1)
imagesc(abs(I))
colormap(gca,'copper')
axis square
colorbar
title('full image')

% spectrum of the full image
subplot(2,3,2)
imagesc(-Y/2,-X/2,log(abs(fulldata)))
colormap(gca,'gray')
axis square
hold on
scatter(X*kx,Y*ky)
title('spectrum of the full image')

% spectrum of the full image along the radial
subplot(2,3,3)
plot((-Nsamples/2):(Nsamples/2-1), log(abs(partdata)))
title('spectrum of the full image along the radial')

% subsampled image
subplot(2,3,4)
imagesc(abs(subI))
colormap(gca,'copper')
axis square
colorbar
title('subsampled image')

% spectrum of the subsampled image
subplot(2,3,5)
imagesc(-subY/2,-subX/2,log(abs(subfulldata)))
colormap(gca,'gray')
axis square
hold on
scatter(subX*subkx,subY*subky)
title('spectrum of the subsampled image')

% spectrum of the subsampled image along the radial
subplot(2,3,6)
semilogy((-Nsamples/2):(Nsamples/2-1), abs(subpartdata))
hold on
semilogy((-Nsamples/2):(Nsamples/2-1), abs(partdata)/factor^2)
legend('spectrum of the subsampled image along the radial',...
       'scaled spectrum of the full image along the radial (for reference)')
title('spectrum of the subsampled image along the radial')