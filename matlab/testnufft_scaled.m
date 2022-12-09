% the script performs the comparison of NUFFT by Fessler with the linear
% interpolation after fft2, and restores the image as a subsamlped one
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

% choose the k-space sampling pattern
sampling = 1; % 1: radial, 2: cartesian

% pre-set parameters
if sampling == 1
    Nsamples = 128; % number of acquired samples per each radial line
    rads     = 128; % number of radial lines
else
    xsamples = 128;
    ysamples = 128;
end
X = 1024;   % image width
Y = 1024;   % image height
factor = 4; % factor of width and height reduction in the reconstruction

%% generate an image
I = imread([datafold, '/panda.png']);
I = imresize(I, [Y-256 X-256]);
I = rgb2gray(I);
newI = zeros(Y,X);
newI(129:end-128,129:end-128) = I;
I = newI;
I = double(I);
I = I/max(I(:)) + 0.5i*rand(Y,X);

%% simulate the acquisition
% the acquisition does not use any density compensation, as done in the
% simulator
if sampling == 1
    golden = 2*pi /(1+sqrt(5));
    angles = (0:rads-1)*golden;

    % these are the acquired locations in the k-space
    kx = ( (-Nsamples/2):(Nsamples/2-1) )' * cos(angles);
    ky = ( (-Nsamples/2):(Nsamples/2-1) )' * sin(angles);
else
    [kx, ky] = meshgrid( (-xsamples/2):(xsamples/2-1), (-ysamples/2):(ysamples/2-1) );
end
kx = kx(:)/X;
ky = ky(:)/Y;

% the function DoCalcDCF has been edited such that the areas are not
% normalized
w = DoCalcDCF(ky, kx)*X*Y;

% define the nufft operator
FT = NUFFT(ky + 1i*kx, 1, [0 0], [Y X]);

% compute the linear interpolation of the spectrum
[XX,YY]   = meshgrid(1:X,1:Y);
XX        = XX - X/2 - 1;
YY        = YY - Y/2 - 1;
fulldata  = fftshift(fft2(fftshift(I)));
partdata  = interp2(XX,YY,fulldata,X*kx,Y*ky,'linear');
nufftdata = sqrt(X*Y)*(FT*I); % scaled to resemble the interpolated data
    
%% compute the reconstruction operator
newX = X/factor;
newY = Y/factor;
if sampling == 1
    golden = 2*pi /(1+sqrt(5));
    angles = (0:rads-1)*golden;

    % these are the acquired locations in the k-space
    kx = ( (-Nsamples/2):(Nsamples/2-1) )' * cos(angles);
    ky = ( (-Nsamples/2):(Nsamples/2-1) )' * sin(angles);
else
    [kx, ky] = meshgrid( (-xsamples/2):(xsamples/2-1), (-ysamples/2):(ysamples/2-1) );
end
kx = kx(:)/newX;
ky = ky(:)/newY;

% the function DoCalcDCF has been edited such that the areas are not
% normalized
neww = DoCalcDCF(ky, kx)*newX*newY;

% define the nufft operator
newFT = NUFFT(ky + 1i*kx, neww, [0 0], [newY newX]);

%% manipulate the data
partdata = partdata.*sqrt(neww)'; % pre-multiply with the weights
nufftdata = nufftdata.*sqrt(neww)'; % pre-multiply with the weights

%% compute the inverse (nu)fft
fullreco = fftshift(ifft2(fftshift(fulldata)));
acqreco1 = newFT'*partdata/(sqrt(X*Y)*factor);
acqreco2 = newFT'*nufftdata/(sqrt(X*Y)*factor);
acqreco3 = newFT'*(newFT*imresize(I,1/factor));

%% plot everything
figure
subplot(2,2,1)
imagesc(abs(fullreco))
title('full fft + ifft')
axis square
colormap('copper')
colorbar

subplot(2,2,2)
imagesc(abs(acqreco1))
title('interpolated fft + inverse nufft + scaling')
axis square
colormap('copper')
colorbar

subplot(2,2,3)
imagesc(abs(acqreco2))
title('nufft + inverse new nufft + scaling')
axis square
colormap('copper')
colorbar

subplot(2,2,4)
imagesc(abs(acqreco3))
title('new nufft + inverse new nufft')
axis square
colormap('copper')
colorbar