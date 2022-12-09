% the script performs the comparison of different methods to compute the
% non-uniform Fourier transform:
% (1) linear interpolation
% (2) NUFFT by Fessler
% (3) NUFFT separately per each radial
% (4) NUFFT separately per each coefficient
% (5) dtft by Fessler
% (6) explicit computation of the non-uniform Fourier transform
%
% Date: 10/05/2021
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

addpath('reconstruction')
rng(0)

% pre-set parameters
Nsamples = 128;  % number of acquired samples per each radial line
X        = 1024; % image width
Y        = 1024; % image height
rads     = 5;    % number of radial lines

%% generate an image
margX = X/4;
margY = Y/4;
I = imread([datafold, '/panda.png']);
I = imresize(I, [Y-margY, X-margX]);
I = rgb2gray(I);
newI = zeros(Y,X);
newI(margY/2+1:end-margY/2,margX/2+1:end-margX/2) = I;
I = newI;
I = double(I);
I = I/norm(I(:)) + 0.0005i*randn(Y,X);

%% simulate the acquisition
golden = 2*pi /(1+sqrt(5));
angles = (0:rads-1)*golden;

% these are the acquired locations in the k-space
kx = ( (-Nsamples/2):(Nsamples/2-1) )' * cos(angles);
ky = ( (-Nsamples/2):(Nsamples/2-1) )' * sin(angles);
kx = kx/X;
ky = ky/Y;

% compute the density compensation
% the function DoCalcDCF has been edited such that the areas are not
% normalized
w = DoCalcDCF(kx(:), ky(:))' * X*Y;
w = reshape(w,Nsamples,rads);

% compute the linear interpolation of the spectrum
[XX,YY]  = meshgrid(1:X,1:Y);
XX       = XX - X/2 - 1;
YY       = YY - Y/2 - 1;
fulldata = fftshift(fft2(fftshift(I)));
partdata = sqrt(w).*interp2(XX,YY,fulldata,X*kx,Y*ky,'linear');
partdata = partdata(:);

% compute the nufft
FT = NUFFT(ky(:) + 1i*kx(:), w(:), [0 0], [Y X]);
nufftdata1 = sqrt(X*Y)*(FT*I);

% compute the nufft per radial
nufftdata2 = NaN(Nsamples,rads);
for radial = 1:rads
    FT = NUFFT(ky(:,radial) + 1i*kx(:,radial), w(:,radial), [0 0], [Y X]);
    nufftdata2(:,radial) = FT*I;
end
nufftdata2 = sqrt(X*Y)*nufftdata2(:);

% compute the nufft per coefficient
nufftdata3 = NaN(Nsamples,rads);
for radial = 1:rads
    for sample = 1:Nsamples
        FT = NUFFT(ky(sample,radial) + 1i*kx(sample,radial), w(sample,radial), [0 0], [Y X]);
        nufftdata3(sample,radial) = FT*I;
    end
end
nufftdata3 = sqrt(X*Y)*nufftdata3(:);

% compute the dtft by Fessler
dtftdata = NaN(Nsamples,rads);
for radial = 1:rads
    dtftdata(:,radial) = dtft(I, 2*pi*[ky(:,radial) kx(:,radial)], [Y/2 X/2], false);
end
dtftdata = sqrt(w(:)).*dtftdata(:);

% compute the dtft by hand
handdata = NaN(Nsamples,rads);
for radial = 1:rads
    for sample = 1:Nsamples
        handdata(sample,radial) = sum(I .* exp(-2*pi*1i*kx(sample,radial)*XX) .* exp(-2*pi*1i*ky(sample,radial)*YY),'all');
    end
end
handdata = sqrt(w(:)).*handdata(:);

%% plot
radial = randi(rads);
plotdata = [partdata,...
    nufftdata1,...
    nufftdata2,...
    nufftdata3,...
    dtftdata,...
    handdata ];
indices = (radial-1)*Nsamples+1:radial*Nsamples;

figure

% real part of selected coefficients
subplot(2,2,1)
plot(real(plotdata(indices,:)))
legend('interpolation','nufft','nufft per radial','nufft per coef','dtft','by hand')
title('real part')

% imaginary part of selected coefficients
subplot(2,2,3)
plot(imag(plotdata(indices,:)))
legend('interpolation','nufft','nufft per radial','nufft per coef','dtft','by hand')
title('imaginary part')

% differences
subplot(2,2,[2 4])
semilogy(abs(plotdata(indices,1:5) - repmat(plotdata(indices,6),1,5)))
legend('interpolation','nufft','nufft per radial','nufft per coef','dtft')
title('absolute difference to by hand')

sgtitle(['radial number ',num2str(radial)])

%% command window output
fprintf('Maximum absolute differences of the coefficients\n')
fprintf('  (1) linear interpolation\n')
fprintf('  (2) NUFFT by Fessler\n')
fprintf('  (3) NUFFT separately per each radial\n')
fprintf('  (4) NUFFT separately per each coefficient\n')
fprintf('  (5) dtft by Fessler\n')
fprintf('  (6) explicit computation of the non-uniform Fourier transform\n\n')
dists = NaN(size(plotdata,2));
for i = 1:size(plotdata,2)
    for j = 1:size(plotdata,2)
        dists(i,j) = max(abs(plotdata(:,i)-plotdata(:,j)));
    end
end
disp(dists)