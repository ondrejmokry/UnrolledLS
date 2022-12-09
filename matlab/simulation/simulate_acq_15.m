% Use this single script to generate synthetic DCE-MRI data

% The current version is adapted to the needs of the KInG project, e.g., it
% generates alse the ground truth sequences with predefined time and space
% resolution

clear
clc
close all
rng(0)

if(isfile('../../custom_config_simulator.json'))
    fid = fopen('../../custom_config_simulator.json'); % Opening the file
else
    fid = fopen('../../default_config_simulator.json'); % Opening the file
end
raw = fread(fid,inf); % Reading the contents
str = char(raw'); % Transformation
fclose(fid);

hyper_parameters = jsondecode(str);

% add the package
fileID = fopen(hyper_parameters.packagefold,'r');
packagefold = fscanf(fileID,'%s');
fclose(fileID);
addpath(genpath([packagefold, '/ESPIRiT']));

fileID = fopen(hyper_parameters.datafold,'r');
df = fscanf(fileID,'%s');
fclose(fileID);

%% input file name
% shuldn't be changed; includes ROIs and the corresponding perf. parameters
inpname = [df, '/simulation/mapy.mat_a_tis_170810.mat'];

%% coil sensitivities
% taken from the same recording, preconstrast frames, heavily smoothed
sensitivities = [df, '/simulation/sensitivities.mat'];

% resize factor
ResizeFactor = hyper_parameters.ResizeFactor;

%% output file name and path
[pathstr, name, ext] = fileparts(inpname);
outname = fullfile(pathstr, 'SyntheticEchoes');

%% some global parameters
% radials per frame in the future reconstruction
rpf = hyper_parameters.rpf;

% subsampling in the space dimensions in the future reconstruction
factor = hyper_parameters.factor;

% image oversampling factor
im_ovs = hyper_parameters.im_ovs;

% ignore sensitivities?
no_sens = hyper_parameters.no_sens;

% k-space radial interpolation options
sampling = hyper_parameters.sampling;
switch sampling
    % (1) linear_interp ...... linear interpolations
    case 1, k_space_sampling = 'linear_interp';
    % (2) nufft .............. nufft for whole projection
    case 2, k_space_sampling = 'nufft';
end
outname = [outname, '_', datestr(clock,30)];

% Create acquisition scheme:
% list of acquisition steps, for each TR (i.e. for each echo, or set of
% echoes for array coils) the "acquired" echo signal is specified as one
% entry in the following two vectors:
%   Angles - angle of the acquisition trajectory line [rad]
%   PhEncShifts - shift in the phase encoding direction [samples in the k-space]

% Radial 2D golden angle sampling:
Projections = hyper_parameters.Projections; % affects acquisition length (= Projections * TR) should be at least cca 10 min (standard duration: 10 - 15 min)
GoldenAngle = 2*pi /(1+sqrt(5)); % [rad]
Angles = (0:(Projections-1)) * GoldenAngle; % angles using golden angle increment [rad]
PhEncShifts = zeros(1,Projections); % phase encoding shifts - for radial sampling always zero

%% global tissue parameters
r1  = hyper_parameters.r1;  % r1 relaxivity of tissue and blood [l/mmol/s]
R10 = hyper_parameters.R10; % 1/s

%% acquisition parameters
TR = hyper_parameters.TR;     % TR [s] (time interval between radial acquisition)
FA = hyper_parameters.FA*pi/180; % flip angle of excitation RF pulses [rad]
Nsamples = hyper_parameters.Nsamples; % number of samples of each echo (sampling rate is fixed, given by the FOV)
delay = hyper_parameters.delay;     % bolus arrival time [s]

krho = hyper_parameters.krho; % multiplication factor accounting for spin density, gains,... here set arbitrarily, don't change (it would change the SNR which is now tuned to agree with real data)
% SD = 0.14; % standard deviation of noise
SD = 0.25;

%% time axis in [min]
% temporal sampling period for generation of the curves [s], now equal to
% TR, shouldn't be changed
Ts = TR;

% number of frames (number of temporal samples) of the concentration curves
N = length(Angles);

% time axis [min]
t = (0:(N-1))*Ts/60;

% minimum length for zeropadding
Nmin = N*2-1;

% the closest power of two
NN = ( 2^ceil(log2(Nmin)) );

%% AIF
A    = 2.254;
B    = 0.8053;
C    = 0.5381;
tau1 = 1.433;
tau2 = 2.6349;
tau3 = 0.07;
beta = 0.0847;
aif  = AIF_triexpG(A, B, C, tau1, tau2, tau3, beta, Ts/60, N);
delay_samples = round(delay / Ts);
aif  = [zeros(1,delay_samples) aif(1:(N-delay_samples))]; % delay AIF

%% load input data
load(inpname)
NumOfRois = size(tissue,1);

%% modify input geometry
xsize   = size(data2{1},2);
ysize   = size(data2{1},1);
testmap = zeros(ysize,xsize);

for roi = 1:NumOfRois % for each ROi
    map = full(tissue{roi,2});
    if ResizeFactor == 1
        tissue{roi,2} = logical(map); %#ok<*SAGROW>
        testmap = testmap + map;
    else
        newmap = imresize(map,ResizeFactor,'nearest');
        map = zeros(size(map));
        yi  = int32(round(ysize-size(newmap,1))/2);
        xi  = int32(round(xsize-size(newmap,2))/2);

        % newmap is placed in the middle with respect to the original map
        map(yi:yi+size(newmap,1)-1, xi:xi+size(newmap,2)-1) = newmap;
        tissue{roi,2} = logical(map);
        testmap = testmap + map;
    end

end

%% generate time curves
% time curve for each ROI (and reference perfusion-parameter maps)
Fp = zeros(ysize,xsize);
E  = zeros(ysize,xsize);
ve = zeros(ysize,xsize);
Tc = zeros(ysize,xsize);

for roi = 1:NumOfRois
    fprintf('ROI %2d / %d ', roi, NumOfRois)
    parametry = [tissue{roi,1}(1); tissue{roi,1}(2); tissue{roi,1}(3); tissue{roi,1}(4)] ;
    Fp(tissue{roi,2}) = parametry(1)*(0.6*rand(1)+0.7);
    E(tissue{roi,2})  = parametry(2)*(0.6*rand(1)+0.7);
    ve(tissue{roi,2}) = parametry(3)*(0.6*rand(1)+0.7);
    Tc(tissue{roi,2}) = parametry(4)*(0.6*rand(1)+0.7);
    TRF = DCATH_v5_complete(parametry,t);
    tic
    conc = Ts/60 * real(ifft(   fft(TRF,NN) .* fft(aif,NN)   ));
    conc = conc(1:int32(N));
    R1 = conc*r1+R10; % R1(t) curve
    toc

    tissue{roi,4} = tissue{roi,1};
    tissue{roi,1} = R1;

end
vp     = Tc.*Fp;
Ktrans = Fp .* E;
kep    = E .* Fp ./ ve;
PS     = -log(1-E) .* Fp * (1-0.7*0.4);

%% generate echo signals
load(sensitivities)

if no_sens
    Sensitivities = ones(size(Sensitivities,1),size(Sensitivities,2));
end

CoilElements = size(Sensitivities,3);

echoes = length(Angles); % total number of echoes (TRs)
Nsamples = im_ovs*Nsamples;
EchoSignals = zeros(Nsamples,echoes,CoilElements);

% k-space coordinates
kx = ( (-Nsamples/2):(Nsamples/2-1) )' * cos(Angles);
ky = ( (-Nsamples/2):(Nsamples/2-1) )' * sin(Angles) + repmat(PhEncShifts,[Nsamples, 1]);

% normalization to [-0.5, 0.5]
kx = kx/Nsamples;
ky = ky/Nsamples;

% second normalization to 0.5*[-Nsamples/xsize, Nsamples/xsize]
kx = kx*Nsamples/xsize*0.5*im_ovs;
ky = ky*Nsamples/ysize*0.5*im_ovs;

% saving the coordinates as complex numbers
k = complex(ky,kx); % ugly trick for NUFFT

% initialize images
img_num = floor(echoes/rpf);
img_cnt = 1;
outy = round(ysize/factor);
outx = round(xsize/factor);
imgs = zeros(outy,outx,img_num);

Sens_reco = Sensitivities;

%% fast echo generation
% by Jiri Vitous 2022
% NUFFT source https://github.com/marcsous/nufft_3d
om = [real(k(:))*ysize,imag(k(:))*xsize,zeros(size(k(:)))]';
fprintf('Generating data:\n')
nsamp = Nsamples*Nsamples;
tic
switch k_space_sampling
    case 'nufft'
        obj = nufft_3d(om,[ysize,xsize,1]');
        for coil = 1:CoilElements
            for roi = 1:NumOfRois
                fprintf('Coil: %d/%d, ROI: %d/%d\n',coil,CoilElements,roi,NumOfRois)
                Data_out = obj.fNUFT(tissue{roi,2}.*Sens_reco(:,:,coil));
                Data_out = reshape(Data_out,size(k));
                R1 = tissue{roi,1};
                SI = krho*sin(FA) * (1-exp(-R1*TR)) ./ (1-cos(FA)*exp(-R1*TR));
                SI = repmat(SI,size(Data_out,1),1);
                Data_out = Data_out.*SI;
                EchoSignals(:,:,coil) = EchoSignals(:,:,coil) + Data_out;
            end
        end
    case 'cartesian'
        for coil = 1:CoilElements
            for roi = 1:NumOfRois
                fprintf('Coil: %d/%d, ROI: %d/%d\n',coil,CoilElements,roi,NumOfRois)
                image = imresize(tissue{roi,2}.*Sens_reco(:,:,coil),[Nsamples Nsamples],'Method','nearest');
                Data_out = fftshift(fft2(image/nsamp));
                Data_out = repmat(Data_out,1,ceil(echoes/Nsamples));
                Data_out = Data_out(:,1:echoes);
                R1 = tissue{roi,1};
                SI = krho*sin(FA) * (1-exp(-R1*TR)) ./ (1-cos(FA)*exp(-R1*TR));
                SI = repmat(SI,size(Data_out,1),1);
                Data_out = Data_out.*SI;
                EchoSignals(:,:,coil) = EchoSignals(:,:,coil) + Data_out;
            end
        end
end

% add noise
noise = SD.*randn(size(EchoSignals)) + 1i*SD.*randn(size(EchoSignals));
EchoSignals = EchoSignals + noise;

% create ground truth
fprintf('Generating ground truth:\n')
idx = 1:rpf:echoes;
D = parallel.pool.DataQueue;
afterEach(D, @update_p);
update_p(t,length(idx));

parfor echo = 1:length(idx)
    img = zeros(ysize,xsize);
    for roi = 1:NumOfRois  % for each ROi

        % compute SI (signal intensity) of the given ROI and time instant
        R1 = tissue{roi,1}(idx(echo)); %#ok<PFBNS>
        SI = krho*sin(FA) * (1-exp(-R1*TR)) ./ (1-cos(FA)*exp(-R1*TR));
        img(tissue{roi,2}) = SI;

    end
    send(D,1);
    if(strcmp(k_space_sampling,'nufft'))
        imgs(:,:,echo) = imresize(img,[outy outx]);
    else
        imgs(:,:,echo) = imresize(img,[outy outx],'nearest');
    end
    
end
Sensitivities = imresize(Sensitivities,[outy outx]);

%% save echo signals
save(outname,...
    'Angles',...
    'EchoSignals',...
    'factor',...
    'imgs',...
    'k_space_sampling',...
    'rpf',...
    'r1',...
    'R10',...
    'SD',...
    'Sensitivities',...
    'TR')
toc

function [] = update_p(~,t)
    persistent TOTAL COUNT
    if nargin == 2
        TOTAL = t;
        COUNT = 0;
    else
        COUNT = 1 + COUNT;
        fprintf(['%d/%d Ground truths'  '\n'],COUNT,TOTAL);
   end
end