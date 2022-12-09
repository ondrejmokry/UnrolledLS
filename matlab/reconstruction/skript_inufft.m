% the script performs reconstruction from the synthetic DCE-MRI data from
% the simulator using only the NUFFT operator
%
% Date: 02/06/2021
% By Ondrej Mokry
% Brno University of Technology
% Contact: ondrej.mokry@mensa.cz

clear
clc
close all

% add the package
fileID = fopen('../../packagefold.txt','r');
packagefold = fscanf(fileID,'%s');
fclose(fileID);
addpath(genpath([packagefold, '/ESPIRiT']));

fileID = fopen('../../datafold.txt','r');
df = fscanf(fileID,'%s');
fclose(fileID);

rng(0)

%% load the data
datafold = [df, '/simulation/'];
datafile = 'SyntheticEchoes_nufft_64';
recofold = [df, '/reconstruction/'];

load([datafold, datafile])
load([datafold, 'sensitivities'])

[Nsamples, ~, Ncoils] = size(EchoSignals);
[Y, X, ~] = size(Sensitivities);
% aif           perfusion curve in N time instants (vector 1 x N)
% Angles        angles under which the data are sampled [rad] (vector 1 x N)
% delay         bolus arrival time [s]
% EchoSignals   the signal (array Nsamples x N x Ncoils)
% FA            flip angle of excitation RF pulses [rad]
% N             number of samples
% Ncoils        number of coil elements
% Nsamples      number of samples of each echo
% PhEncShifts   phase encoding shifts, for radial sampling always zero (vector 1 x N)
% r1            r1 relaxivity of tissue and blood [l/mmol/s]
% Sensitivities sensitivities of the coil elements (array X x Y x Ncoils)
% TR            time interval between two consecutive samples [s]
% X, Y          size of the image

%% crop the beginning
startpoint  = 20; % crop interval [s]
cropsamples = ceil(startpoint/TR);
aif         = aif(cropsamples+1:end);
Angles      = Angles(cropsamples+1:end);
EchoSignals = EchoSignals(:,cropsamples+1:end,:);
N           = N - cropsamples;
PhEncShifts = PhEncShifts(cropsamples+1:end);

%% settings
% radials per frame
rpf = 37;

% number of frames
Nf = 100; % number of frames
Nf = min(Nf, floor(N/rpf));

% subsampling factor (for speed-up)
factor = 4;

%% subsample and normalize the sensitivities
% subsample
X = X/factor;
Y = Y/factor;
C = zeros(Y,X,Ncoils);
for c = 1:Ncoils
    C(:,:,c) = imresize(Sensitivities(:,:,c), 1/factor);
end

% normalize
norms = sqrt(sum(conj(C).*C,3));
C = C./norms;

%% precompute the NUFFT operators
w   = zeros(Nsamples*rpf,Nf);
FFT = cell(Nf,1);
for f = 1:Nf
    
    fprintf('Defining the NUFFT for frame %d of %d.\n',f,Nf)
    
    % compute the k-space locations of the samples (taking into account the
    % convention of NUFFT)
    fAngles = Angles((f-1)*rpf+1:f*rpf);
    kx      = ( (-Nsamples/2):(Nsamples/2-1) )'*sin(fAngles);
    ky      = ( (-Nsamples/2):(Nsamples/2-1) )'*cos(fAngles);
    kx      = kx(:)/X;
    ky      = ky(:)/Y;
    
    % compute the density compensation
    % the function DoCalcDCF has been edited such that the areas are not
    % normalized
    w(:,f) = DoCalcDCF(kx, ky)*X*Y;
    
    % compute the NUFFT
    FFT{f} = NUFFT(kx + 1i*ky, w(:,f), [0 0], [Y X]);
end

%% reorganize the data
% the goal is to have it in an array of size length(NUFFT) x Nf * Ncoils
% where length(NUFFT) = rpf * Nsamples
d = EchoSignals(:,1:rpf*Nf,:);
d = reshape(d,[rpf*Nsamples,Nf,Ncoils]);

% density compensation
for c = 1:Ncoils  
    d(:,:,c) = d(:,:,c).*sqrt(w);   
end

% data normalization
d = d/sqrt(X*Y);

%% compute the inverse NUFFT of the data
% compute
simple = zeros(Y,X,Nf);
for f = 1:Nf
    
    fprintf('Reconstructing frame %d of %d.\n',f,Nf)
    
    for c = 1:Ncoils
        simple(:,:,f) = simple(:,:,f) + conj(C(:,:,c)).*(FFT{f}'*d(:,f,c));
    end
end

% choose the frame to plot
plotframe = randi(Nf);

%% compute the ground truth
fprintf('Generating the ground truth.\n')
addpath('../simulation')
gt = getImages(startpoint + (rpf*TR) * (0:Nf-1), 1/factor);

%% compute the version multiplied with the sensitivities
gtc = gt;
for f = 1:Nf
    gtc(:,:,f) = gtc(:,:,f).*sqrt(sum(conj(C).*C,3));
end

%% plot the perfusion curves
figure

% prepare the time axis
timeperframe = rpf * TR;
timeaxis = cropsamples * TR + (0 : Nf - 1)*timeperframe;

% save the data
[~,~] = mkdir(recofold);
save([recofold, 'INUFFT_', datestr(clock,30)],...
    'TR', 'rpf', 'Nf', 'cropsamples', 'timeaxis', 'simple', 'gt', 'gtc')

% reconstructed frame
s = subplot(2,3,1);
imagesc(abs(simple(:,:,plotframe)))
hold on
colorbar
axis square
title({sprintf('reconstructed frame %d of %d',plotframe,Nf),...
    'right-click to show a perfusion curve, close the figure to end'})

% ground truth frame
subplot(2,3,2);
imagesc(abs(gt(:,:,plotframe)))
hold on
colorbar
axis square
title(sprintf('ground truth frame %d of %d',plotframe,Nf))

% ground truth frame multiplied with the sensitivity norms
subplot(2,3,3);
imagesc(abs(gtc(:,:,plotframe)))
hold on
colorbar
axis square
title({sprintf('ground truth frame %d of %d',plotframe,Nf),...
    'multiplied with the sensitivity norms'})

% perfusion curve
while 1
    
    try
        [x, y] = ginput(1);
        x = round(x);
        y = round(y);
        x = max(1,min(x,X));
        y = max(1,min(y,Y));
    catch
        fprintf('Mischief managed!\n')
        break
    end
    imagesc(s,abs(simple(:,:,plotframe)))
    scatter(s,x,y,'r')
    
    subplot(2,3,4:6)
    yyaxis left
    plot(timeaxis,squeeze(abs(simple(y,x,:))),...
         timeaxis,squeeze(abs(gtc(y,x,:))),':')    
    yyaxis right
    plot(timeaxis,squeeze(abs(gt(y,x,:))))
    xlabel('time / s')
    title(sprintf('perfusion curve at [%d,%d]',x,y))
    legend('inverse NUFFT','ground truth times sensitivity norms','ground truth')
    ax = gca;
    ax.XGrid = 'on';
    ax.XMinorGrid = 'on';
    xlim([min(timeaxis), max(timeaxis)])
    
end