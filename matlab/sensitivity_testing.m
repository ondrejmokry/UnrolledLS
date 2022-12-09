% the script performs the comparison of pre-set sensitivity maps with the
% maps estimated via ESPIRiT
%
% it shows that the estimated sensitivities closely resemble the pre-set
% maps with the only major difference that the estimated maps are
% normalized per each image pixel
%
% the ESPIRiT procedure and visualization is taken from the ESPIRiT Maps
% Demo (demo_ESPIRiT_maps.m) which comes with the ESPIRiT Matlab package
%
% if part one is not skipped, it also demonstrates the low-frequency nature
% of the pre-set sensitivities by low pass filtering, suggesting that
% subsampling of the sensitivities will not produce unreasonable maps
%
% Date: 13/01/2021
% By Ondrej Mokry
% Brno University of Technology
% Contact: ondrej.mokry@mensa.cz

clear
close all
clc

map = 'copper';

skippartone = false;

% add the package
fileID = fopen('../packagefold.txt','r');
packagefold = fscanf(fileID,'%s');
fclose(fileID);
addpath(genpath([packagefold, '/ESPIRiT']));

fileID = fopen('../datafold.txt','r');
datafold = fscanf(fileID,'%s');
fclose(fileID);

%% demonstration of the low-frequency nature of the sensitivities
% load the sensitivities
load([datafold, '/simulation/sensitivities.mat'])

% plot the sensitivities
[Y, X, C] = size(Sensitivities);

if ~skippartone
    figure
    sgtitle('sensitivities')
    colormap(map)
    for c = 1:C

        % real part
        subplot(3,C,c)
        imagesc(real(Sensitivities(:,:,c)))
        axis square
        axis off
        title(['coil ', num2str(c), ', real part'])
        colorbar

        % imaginary part
        subplot(3,C,C+c)
        imagesc(imag(Sensitivities(:,:,c)))
        axis square
        axis off
        title(['coil ', num2str(c), ', imaginary part'])
        colorbar

        subplot(3,C,2*C+c)
        imagesc(abs(Sensitivities(:,:,c)))
        axis square
        axis off
        title(['coil ', num2str(c), ', absolute value'])
        colorbar
    end

    % filter the sensitivities with a LPF and evaluate the difference
    [ Xs, Ys ] = meshgrid((-X/2 : X/2 - 1)/X, (-Y/2 : Y/2 - 1)/Y);
    distance   = sqrt(Xs.^2 + Ys.^2);
    factors    = 2.^(1:5);
    counter    = 1;
    figure
    sgtitle('abs. difference after LPF')
    colormap(map)
    for c = 1:C
        FFT = fftshift(fft2(Sensitivities(:,:,c)));
        for factor = factors

            % compute the LPF
            newFFT = FFT;
            newFFT(distance > 1/(2*factor)) = 0;
            decimated = ifft2(ifftshift(newFFT));

            % plot the result
            subplot(C, length(factors), counter)
            imagesc(abs(Sensitivities(:,:,c)-decimated))
            axis square
            axis off
            title(sprintf('coil %d, dec. factor %d',c,round(factor)))
            colorbar

            counter = counter + 1;
        end
    end
end

%% comparison of some pre-set sensitivities and ESPIRiT
resizefactor = 1/4;
for i = 1:2
    
    % generate an image + its mask
    if i == 1
        I = imread([datafold, '/panda.png']);
        I = imresize(I, [Y-256 X-256]);
        I = rgb2gray(I);
        newI = zeros(Y,X);
        newI(129:end-128,129:end-128) = I;
        I = newI;
        mask = I > 0;
        I = double(I);
        I = I/max(I(:));
    else
        load([datafold, '/img.mat']);
        I = img;
        mask = testmap;
    end
    I             = imresize(I, resizefactor*[Y, X]);
    mask          = imresize(mask, resizefactor*[Y, X]);
    Sensitivities = imresize(Sensitivities, resizefactor*[Y, X]);
    
    % simulate the (fully sampled) acquisition
    data = zeros(resizefactor*Y,resizefactor*X,C);
    for c = 1:C
        data(:,:,c) = fftshift(fft2(ifftshift(I.*Sensitivities(:,:,c))));
        data(:,:,c) = data(:,:,c)/sqrt(resizefactor*Y * resizefactor*X);
    end
    
    % display coil images
    im = ifft2c(data);
    
    % the coil images computed via ifft2c are equal to the coil images
    % obtained as I.*Sensitivities(:,:,c)

    figure
    subplot(2,1,1)
    imshow3(abs(im),[],[1,C]); 
    title('magnitude of physical coil images');
    colormap(map); colorbar;

    subplot(2,1,2)
    imshow3(angle(im),[],[1,C]); 
    title('phase of physical coil images');
    colormap(map); colorbar;

    % compute the sensitivities using ESPIRiT  
    ncalib = 24; % use 24 calibration lines to compute compression
    ksize  = [6, 6];
    eigThresh_1 = 0.02;
    eigThresh_2 = 0.95;
    calib  = crop(data,[ncalib,ncalib,C]);
    [k, S] = dat2Kernel(calib,ksize);    
    idx    = find(S >= S(1)*eigThresh_1, 1, 'last');   
    [M, W] = kernelEig(k(:,:,:,1:idx),[resizefactor*Y, resizefactor*X]);
    kdisp  = reshape(k,[ksize(1)*ksize(2)*C,ksize(1)*ksize(2)*C]);
    
    % this shows that the calibration matrix has a null space
    figure
    subplot(2,1,1)
    plot(1:ksize(1)*ksize(2)*C,S);
    hold on
    plot([1,ksize(1)*ksize(2)*C],[S(1)*eigThresh_1,S(1)*eigThresh_1]);
    plot([idx,idx],[0,S(1)])
    legend('signular vector value','threshold')
    title('singular vectors')
    subplot(212), imagesc(abs(kdisp)), colormap(map);
    title('singular vectors')
    
    % show eigen-values and eigen-vectors, the last set of eigen-vectors
    % corresponding to eigen-values 1 look like sensitivity maps
    figure
    subplot(3,4,1:4)
    imshow3(abs(W),[],[1,C]); 
    title('eigen values in image space');
    colormap(map); colorbar;

    subplot(3,4,[5 6 9 10])
    imshow3(abs(M),[],[C,C]); 
    title('magnitude of eigen vectors');
    colormap(map); colorbar;

    subplot(3,4,[7 8 11 12])
    imshow3(angle(M),[],[C,C]); 
    title('phase of eigen vectors');
    colormap(map); colorbar;
    
    % project onto the eigenvectors
    % this shows that all the signal energy lives in the subspace spanned
    % by the eigenvectors with eigenvalue 1 (these look like sensitivity
    % maps)
    P = sum(repmat(im,[1,1,1,C]).*conj(M),3);
    figure
    subplot(2,1,1)
    imshow3(abs(P),[],[1,C]); 
    title('magnitude of the coil images projected onto the eigenvectors');
    colormap(map); colorbar;

    subplot(2,1,2)
    imshow3(angle(P),[],[1,C]); 
    title('phase of the coil images projected onto the eigenvectors');
    colormap(map); colorbar;
    
    % alternative way to compute projection is
    % ESP = ESPIRiT(M);
    % P = ESP'*im;
    
    % crop sensitivity maps
    maps = M(:,:,:,end).*repmat(W(:,:,end) > eigThresh_2,[1,1,C]);

    % compare
    figure
    subplot(4,1,1)
    imshow3(abs(Sensitivities),[],[1,C]); 
    title('original sensitivities (magnitude)');
    colormap(map); colorbar;

    subplot(4,1,2)
    imshow3(abs(maps),[],[1,C]); 
    title('estimated sensitivities (magnitude)');
    colormap(map); colorbar;
    
    subplot(4,1,3)
    imshow3(abs(repmat(I,[1,1,C]).*Sensitivities),[],[1,C]);
    title('coil images using original sensitivities (magnitude)');
    colormap(map); colorbar;

    subplot(4,1,4)
    imshow3(abs(repmat(I,[1,1,C]).*maps),[],[1,C]); 
    title('coil images using estimated sensitivities (magnitude)');
    colormap(map); colorbar;
    
    % compare the projection onto the eigenvectors and onto the original
    % maps
    S1norm = Sensitivities./sqrt(sum(conj(Sensitivities).*Sensitivities,3));
    S1norm(abs(Sensitivities) == 0) = 0;
    S2norm = maps./sqrt(sum(conj(maps).*maps,3));
    S2norm(abs(maps) == 0) = 0; 
    im1    = zeros(size(im));
    im2    = zeros(size(im));
    for c = 1:C
        im1(:,:,c) = S1norm(:,:,c).*sum(conj(S1norm).*im,3);
        im2(:,:,c) = S2norm(:,:,c).*sum(conj(S2norm).*im,3);
    end

    figure
    subplot(2,1,1)
    imshow3(abs(im-im1),[],[1,C])
    title('original coil images minus the projection onto the original sensitivities (magnitude)')
    colormap(map); colorbar;
    
    subplot(2,1,2)
    imshow3(abs(im-im2),[],[1,C])
    title('original coil images minus the projection onto the estimated sensitivities (magnitude)')
    colormap(map); colorbar;
    
    % compare once again with the normalized and masked sensitivities
    S1norm = S1norm.*repmat(mask,[1,1,C]);
    S2norm = S2norm.*repmat(mask,[1,1,C]);
    
    figure
    subplot(3,2,1)
    imshow3(abs(S1norm),[],[1,C]);
    title('original point-wise normalized sensitivities (magnitude)');
    colormap(map); colorbar;

    subplot(3,2,3)
    imshow3(abs(S2norm),[],[1,C]);
    title('estimated point-wise normalized sensitivities (magnitude)');
    colormap(map); colorbar;
    
    subplot(3,2,5)
    imshow3(abs(abs(S1norm)-abs(S2norm)),[],[1,C]);
    
    title('absolute difference of magnitudes');
    colormap(map); colorbar;
    
    subplot(3,2,2)
    imshow3(abs(repmat(I,[1,1,C]).*S1norm),[],[1,C]);
    title('coil images using original point-wise normalized sensitivities (magnitude)');
    colormap(map); colorbar;

    subplot(3,2,4)
    imshow3(abs(repmat(I,[1,1,C]).*S2norm),[],[1,C]);
    title('coil images using estimated point-wise normalized sensitivities (magnitude)');
    colormap(map); colorbar;
    
    subplot(3,2,6)
    imshow3(abs(abs(repmat(I,[1,1,C]).*S1norm)-abs(repmat(I,[1,1,C]).*S2norm)),[],[1,C]);
    title('absolute difference of magnitudes');
    colormap(map); colorbar;
    
end