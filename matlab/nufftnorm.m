% the script tries to compute the operator norm of non-uniform Fourier
% transform in a given setting
%
% Date: 15/02/2021
% By Ondrej Mokry
% Brno University of Technology
% Contact: ondrej.mokry@mensa.cz

clear
clc
close all
addpath('rekonstrukce')

% add the package
fileID = fopen('../packagefold.txt','r');
packagefold = fscanf(fileID,'%s');
fclose(fileID);
addpath(genpath([packagefold, '/ESPIRiT']));

fileID = fopen('../datafold.txt','r');
datafold = fscanf(fileID,'%s');
fclose(fileID);

rng(0)

% pre-set parameters
Nsamples = 16; % number of acquired samples per each radial line
X        = 64; % image width
Y        = 64; % image height
Nrads    = 32; % number of radial lines

%% sensitivities
fprintf('Loading, normalizing and subsampling the sensitivities...\n')

% load
load([datafold, '/simulation/sensitivities.mat'])

% normalize and subsample
Sensitivities = Sensitivities./sqrt(sum(conj(Sensitivities).*Sensitivities,3));
Ncoils = size(Sensitivities,3);
C = NaN(Y,X,Ncoils);
for coil = 1:Ncoils
    C(:,:,coil) = imresize(Sensitivities(:,:,coil),[Y, X]);
end
clear Sensitivities

%% generate an image
fprintf('Generating image...\n')
margx = X/4;
margy = Y/4;
I = imread([datafold, '/panda.png']);
I = imresize(I, [Y-margy, X-margx]);
I = rgb2gray(I);
newI = zeros(Y,X);
newI((margy/2 + 1):(end - margy/2), (margx/2 + 1):(end - margx/2)) = I;
I = newI;
I = double(I);
I = I/norm(I(:)) + 0.0005i*randn(Y,X);

%% compute the Fourier transform
fprintf('Generating k-space positions...\n')
golden = 2*pi /(1+sqrt(5));
angles = (0:Nrads-1)*golden;

% these are the acquired locations in the k-space
kx = ( (-Nsamples/2):(Nsamples/2-1) )' * cos(angles);
ky = ( (-Nsamples/2):(Nsamples/2-1) )' * sin(angles);
kx = kx/X;
ky = ky/Y;

% compute the density compensation
fprintf('Generating density weights...\n')
w = DoCalcDCF(kx(:), ky(:))' * X*Y;
w = reshape(w,Nsamples,Nrads);

% compute the dtft by hand
fprintf('Computing dtft for ')
[XX,YY] = meshgrid(1:X,1:Y);
XX      = XX - X/2 - 1;
YY      = YY - Y/2 - 1;
dataEXP = NaN(Nsamples,Nrads,Ncoils); % data per whole frame
for coil = 1:Ncoils
    fprintf(['coil ', num2str(coil), '...'])
    for radial = 1:Nrads
        for sample = 1:Nsamples
            dataEXP(sample,radial,coil) = sum(...
                (C(:,:,coil).*I)...
                .* exp(-2*pi*1i*kx(sample,radial)*XX)...
                .* exp(-2*pi*1i*ky(sample,radial)*YY),...
                'all');
        end
    end
    dataEXP(:,:,coil) = dataEXP(:,:,coil) .* sqrt(w);
    if coil < Ncoils
        fprintf(repmat('\b',1,9))
    else
        fprintf('\n')
    end
end
dataEXP = dataEXP(:) / sqrt(X*Y); % divided to correspond to nufft

%% check with the NUFFT package
fprintf('Defining the NUFFT operator...\n')
dataNUFFT = zeros(Nrads*Nsamples,Ncoils);
FT = NUFFT(ky(:) + 1i*kx(:), w(:), [0 0], [Y X]);
fprintf('Computing NUFFT for ')
for coil = 1:Ncoils
    fprintf(['coil ', num2str(coil), '...'])
    dataNUFFT(:,coil) = FT*(C(:,:,coil).*I);
    if coil < Ncoils
        fprintf(repmat('\b',1,9))
    else
        fprintf('\n')
    end
end
dataNUFFT = dataNUFFT(:);

%% define the operator as a matrix
% Fourier transform as a matrix operation
fprintf('Defining the Fourier transform...\n')
F = zeros(Nsamples*Nrads,X*Y);
for radial = 1:Nrads
    for sample = 1:Nsamples
   
        F(sample + (radial-1)*Nsamples,:) = ...
            vec(exp(-2*pi*1i*kx(sample,radial)*XX)...
             .* exp(-2*pi*1i*ky(sample,radial)*YY)) / sqrt(X*Y);
        
    end
end
F = diag(sqrt(w(:)))*F;

% compute the norm as a matrix norm
fprintf('Computing the operator norm of the Fourier operator...\n')
normF = norm(F);

% compute the norm via power iteration, starting from I
fprintf('Computing the operator norm of the Fourier operator via power iteration...\n')
iterations = 100;
normiter = NaN(iterations, 1);
x = I;
for i = 1:iterations
    y = FT'*(FT*x);
    normiter(i) = max(abs(y(:)));
    x = y/normiter(i);
end

% combining the Fourier transform with the sensitivites and density weights
% and scaling
fprintf('Multipliying the sensitivity and Fourier operators for ')
OP = zeros(Ncoils*Nsamples*Nrads,X*Y);
for coil = 1:Ncoils
    fprintf(['coil ', num2str(coil), '...'])
    OP((coil-1)*Nrads*Nsamples + 1 : coil*Nrads*Nsamples, :) = ...
        F*diag(vec(C(:,:,coil)));
    if coil < Ncoils
        fprintf(repmat('\b',1,9))
    else
        fprintf('\n')
    end
end

% compute the norm
fprintf('Computing the operator norm of the whole operator...\n\n')
normOP = norm(OP);

% check that it does the same as above
dataOP = OP*vec(I);
fprintf('Difference of OP and EXP: %.3e\n',norm(dataOP-dataEXP))
fprintf('Difference of OP and NUFFT: %.3e\n',norm(dataOP-dataNUFFT))
fprintf('Operator norm of OP: %.3f\n',normOP)
fprintf('Maximal (square root of) weight: %.3f\n',max(sqrt(w(:))))
fprintf('Operator norm of F (computed from the matrix): %.3f\n',normF)
fprintf('Operator norm of F (computed using NUFFT): %.3f\n',sqrt(normiter(end)))