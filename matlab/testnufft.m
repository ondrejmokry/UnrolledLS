% the script performs the comparison of NUFFT by Fessler with the linear
% interpolation after fft2, answering the question, why the data produced
% with fft2 and reconstructed using NUFFT produce rotated image
%
% Date: 08/02/2021
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
    rads     = 64; % number of radial lines
else
    xsamples = 128;
    ysamples = 128;
end
X = 1024; % image width
Y = 1024; % image height

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
w = DoCalcDCF(kx, ky)*X*Y;

% define the nufft operator
FT = NUFFT(kx + 1i*ky, w, [0 0], [Y X]);

% compute the linear interpolation of the spectrum
[XX,YY]   = meshgrid(1:X,1:Y);
XX        = XX - X/2 - 1;
YY        = YY - Y/2 - 1;
fulldata  = fftshift(fft2(fftshift(I)));
partdata  = sqrt(w)'.*interp2(XX,YY,fulldata,X*kx,Y*ky,'linear');
nufftdata = sqrt(X*Y)*(FT*I); % scaled to resemble the interpolated data

%% compute the inverse (nu)fft
fullreco = fftshift(ifft2(fftshift(fulldata)));
acqreco1 = FT'*partdata;
acqreco2 = FT'*nufftdata;
acqreco3 = FT'*(FT*I);

%% plot everything
figure
subplot(2,2,1)
imagesc(abs(fullreco))
title('full fft + ifft')
axis square
axis off
colormap('copper')
colorbar

subplot(2,2,2)
imagesc(abs(acqreco1))
title('interpolated fft + inverse nufft')
axis square
axis off
colormap('copper')
colorbar

subplot(2,2,3)
imagesc(abs(acqreco2))
title('nufft + scaling + inverse nufft')
axis square
axis off
colormap('copper')
colorbar

subplot(2,2,4)
imagesc(abs(acqreco3))
title('nufft + inverse nufft')
axis square
axis off
colormap('copper')
colorbar

%% plot the center of the spectrum along specified lines
if sampling == 1
    angle    = 1;
    ind      = Nsamples*(angle-1)+1:Nsamples*angle;
    redline  = [X*kx(ind), Y*ky(ind)];
    blueline = [Y*ky(ind), X*kx(ind)];
    figure

    % the center of the spectrum
    subplot(2,4,[1 2 5 6])
    imagesc(-Y/2,-X/2,log(abs(fulldata)))
    xlim([-0.5 0.5]*Nsamples)
    ylim([-0.5 0.5]*Nsamples)
    colormap('gray')
    axis square
    hold on
    line(redline(:,1), redline(:,2), 'color','r')
    line(blueline(:,1),blueline(:,2),'color','b')
    title('log of the magnitude of the spectrum')

    % the full spectrum along the lines
    subplot(2,4,3)
    plot(log(abs(interp2(XX,YY,fulldata,redline(:,1),redline(:,2),'linear'))),'r')
    title('red line')
    subplot(2,4,7)
    plot(log(abs(interp2(XX,YY,fulldata,blueline(:,1),blueline(:,2),'linear'))),'b')
    title('blue line')

    % the simulated data using interpolation for the given angle
    subplot(2,4,4)
    plot(log(abs(partdata(ind)./sqrt(w(ind))')),'k')
    title('interpolated data')

    % the simulated data using nufft for the given angle
    subplot(2,4,8)
    plot(log(abs(nufftdata(ind)./sqrt(w(ind))')),'k')
    title('nufft data')
end