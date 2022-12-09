function [imgs, k_traj, EchoSignals, perfparams_out, interval, aif, Sensitivities] = simulate_acq_16_DCE_analysis(FA, real_data, aif_zero)
% Use this function to generate synthetic DCE-MRI data

% The current version is adapted to the needs of the KInG project, e.g., it
% generates alse the ground truth sequences with predefined time and space
% resolution

if isfile('../../custom_config_simulator.json')
    fid = fopen('../../custom_config_simulator.json'); % Opening the file
else
    fid = fopen('../../default_config_simulator.json'); % Opening the file
end
raw = fread(fid,inf); % Reading the contents
str = char(raw'); % Transformation
fclose(fid);

hyper_parameters = jsondecode(str);
df = hyper_parameters.datafold;

%% input file name
% shuldn't be changed; includes ROIs and the corresponding perf. parameters
inpname = [df, '/simulation/mapy.mat_a_tis_170810.mat'];

%% coil sensitivities
% taken from the same recording, preconstrast frames, heavily smoothed
sensitivities = [df, '/simulation/sensitivities.mat'];

% resize factor
ResizeFactor = hyper_parameters.ResizeFactor;

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

%% global tissue parameters
r1  = hyper_parameters.r1;  % r1 relaxivity of tissue and blood [l/mmol/s]
R10 = hyper_parameters.R10; % 1/s

%% acquisition parameters
TR = hyper_parameters.TR;     % TR [s] (time interval between radial acquisition)
FA = FA*pi/180; % flip angle of excitation RF pulses [rad]
Nsamples = hyper_parameters.Nsamples; % number of samples of each echo (sampling rate is fixed, given by the FOV)
delay = hyper_parameters.delay;     % bolus arrival time [s]

krho = hyper_parameters.krho; % multiplication factor accounting for spin density, gains,... here set arbitrarily, don't change (it would change the SNR which is now tuned to agree with real data)
SD = hyper_parameters.SD_noise; % standard deviation of noise

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

if aif_zero
    aif = zeros(size(aif));
end

%% load input data
load(inpname,'tissue','data2')
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
sweep = 0.2;
for roi = 1:NumOfRois
    fprintf('ROI %2d / %d\n', roi, NumOfRois)
    parametry = [tissue{roi,1}(1)+(2*sweep*rand(1)-sweep)*tissue{roi,1}(1); min(1,tissue{roi,1}(2)+(2*sweep*rand(1)-sweep)*tissue{roi,1}(2)); min(1,tissue{roi,1}(3)+(2*sweep*rand(1)-sweep)*tissue{roi,1}(3)); tissue{roi,1}(4)+(2*sweep*rand(1)-sweep)*tissue{roi,1}(4)] ;
    
%     Fp(tissue{roi,2}) = parametry(1);
%     E(tissue{roi,2})  = parametery(2);
%     ve(tissue{roi,2}) = min(1,parametry(3)+(2*sweep*rand(1)-sweep)*parametry(3));
%     Tc(tissue{roi,2}) = parametry(4)+(2*sweep*rand(1)-sweep)*parametry(4);
    
    Fp(tissue{roi,2}) = parametry(1);
    E(tissue{roi,2})  = parametry(2);
    ve(tissue{roi,2}) = parametry(3);
    Tc(tissue{roi,2}) = parametry(4);
    
%     vp     = Tc.*Fp;
%     Ktrans = Fp .* E;
%     kep    = E .* Fp ./ ve;
%     PS     = -log(1-E) .* Fp * (1-0.7*0.4);
    
    TRF = DCATH_v5_complete(parametry,t,Ts);
    
    %% simpler model
%     exKtrans=unique(Ktrans);
%     exKtrans(exKtrans==0)=[];
%     
%     exve=unique(ve);
%     exve(exve==0)=[];
%     if(isempty(exve))
%     exve=0;
%     end
%     if(isempty(exKtrans))
%     exKtrans=0;
%     end
%     
%     TRF = Kety_with_derivatives([exKtrans,exve],t);

    %% Conversion
    conc = Ts/60 * real(ifft(   fft(TRF,NN) .* fft(aif,NN)   ));
    conc = conc(1:int32(N));
    R1 = conc*r1 + R10; % R1(t) curve
    
    tissue{roi,4} = tissue{roi,1};
    tissue{roi,1} = R1;
    
end
vp     = Tc.*Fp;
Ktrans = Fp .* E;
kep    = E .* Fp ./ ve;
PS     = -log(1-E) .* Fp * (1-0.7*0.4);

%% generate parametric maps
if real_data
    outy = round(ysize/factor);
    outx = round(xsize/factor);
    perfparams_out.Fp = single(imresize(Fp,[outx,outy],'nearest'));
    perfparams_out.E = single(imresize(E,[outx,outy],'nearest'));
    perfparams_out.ve = single(imresize(ve,[outx,outy],'nearest'));
    perfparams_out.Tc = single(imresize(Tc,[outx,outy],'nearest'));
    perfparams_out.vp = single(imresize(vp,[outx,outy],'nearest'));
    perfparams_out.Ktrans = single(imresize(Ktrans,[outx,outy],'nearest'));
    perfparams_out.kep = single(imresize(kep,[outx,outy],'nearest'));
    perfparams_out.PS = single(imresize(PS,[outx,outy],'nearest'));
else
    perfparams_out = [];
end


%% generate echo signals
% initialize images
outy = round(ysize/factor);
outx = round(xsize/factor);

% load sensitivities
load(sensitivities,'Sensitivities')
if no_sens
    Sensitivities = ones(size(Sensitivities,1),size(Sensitivities,2));
end
CoilElements = size(Sensitivities,3);
Sens_reco = Sensitivities;

% subsample and normalize the sensitivities
Sensitivities = imresize(Sensitivities,[outy outx]);
Sensitivities = Sensitivities./sqrt(sum(conj(Sensitivities).*Sensitivities,3));

% initialize echoes
echoes = length(Angles); % total number of echoes (TRs)
Nsamples = im_ovs*Nsamples;
EchoSignals = zeros(Nsamples,echoes,CoilElements);

% k-space coordinates
kx = ( (-Nsamples/2):(Nsamples/2-1) )' * cos(Angles);
ky = ( (-Nsamples/2):(Nsamples/2-1) )' * sin(Angles);

% normalization to [-0.5, 0.5]*(output_dimension)*im_ovs
kx = kx/Nsamples*outx*im_ovs;
ky = ky/Nsamples*outy*im_ovs;

if real_data
    %% fast echo generation
    % by Jiri Vitous 2022
    % NUFFT source https://github.com/marcsous/nufft_3d
    k_traj = [ky(:),kx(:),zeros(size(kx(:)))]';
    
    fprintf('Generating data:\n')
    nsamp = Nsamples*Nsamples;
    switch k_space_sampling
        case 'nufft'
            obj = nufft_3d(k_traj,[ysize,xsize,1]');
            for coil = 1:CoilElements
                for roi = 1:NumOfRois
                    fprintf('Coil: %d/%d, ROI: %d/%d\n',coil,CoilElements,roi,NumOfRois)
                    Data_out = obj.fNUFT(tissue{roi,2}.*Sens_reco(:,:,coil));
                    Data_out = reshape(Data_out,size(kx));
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
end

%% create ground truth
% by Jiri Vitous 2022
fprintf('Generating ground truth:\n')
full_sampling_cond = 4*pi/2*outx;
Angles_gtc = (0:(full_sampling_cond-1)) * 2*pi/full_sampling_cond;
echoes = length(Angles_gtc); % total number of echoes (TRs)

% k-space coordinates
kx_gtc = ( (-Nsamples/2):(Nsamples/2-1) )' * cos(Angles_gtc);
ky_gtc = ( (-Nsamples/2):(Nsamples/2-1) )' * sin(Angles_gtc);

% normalization to [-0.5, 0.5]*(output_dimension)*im_ovs
kx_gtc = kx_gtc/Nsamples*outx*im_ovs;
ky_gtc = ky_gtc/Nsamples*outy*im_ovs;
om_gtc = [ky_gtc(:),kx_gtc(:),zeros(size(kx_gtc(:)))]';

% initialize images
imgs = zeros(outy,outy,length(1:rpf:length(tissue{roi,1})),CoilElements);
interval = TR*rpf;

simple=0;

switch k_space_sampling
    case 'nufft'
        if(simple==0)
            obj = nufft_3d(om_gtc,[ysize,xsize,1]');
            obj_adj = nufft_3d(om_gtc,[outy,outx,1]');
            for coil = 1:CoilElements
                EchoSignals_gtc = zeros(Nsamples*echoes,length(1:rpf:length(tissue{roi,1})));
                for roi = 1:NumOfRois
                    Data_out = obj.fNUFT(tissue{roi,2}.*Sens_reco(:,:,coil));
                    fprintf('Coil: %d/%d, ROI: %d/%d\n',coil,CoilElements,roi,NumOfRois)
                    R1 = tissue{roi,1};
                    SI = krho*sin(FA) * (1-exp(-R1*TR)) ./ (1-cos(FA)*exp(-R1*TR));
                    SI = SI(1:rpf:end);
                    Data_out = repmat(Data_out,1,length(SI));
                    EchoSignals_gtc = EchoSignals_gtc + Data_out.*SI;
                end
                imgs(:,:,:,coil) = squeeze(obj_adj.iNUFT(EchoSignals_gtc));
            end
        else
            for coil = 1:CoilElements
                EchoSignals_gtc=zeros(outx,outy,length(tissue{1,1}(1:rpf:end)));
                for roi = 1:NumOfRois
                    Data_out = imresize(tissue{roi,2}.*Sens_reco(:,:,coil),[outy,outy],'nearest');
                    fprintf('Coil: %d/%d, ROI: %d/%d\n',coil,CoilElements,roi,NumOfRois)
                    R1 = tissue{roi,1};
                    SI = krho*sin(FA) * (1-exp(-R1*TR)) ./ (1-cos(FA)*exp(-R1*TR));
                    SI = SI(1:rpf:end);
                    SI=permute(SI,[1,3,2]);
                    Data_out = repmat(Data_out,1,1,length(SI));
                    EchoSignals_gtc = EchoSignals_gtc + Data_out.*SI;
                end
                imgs(:,:,:,coil) = EchoSignals_gtc;
            end
            
            
            
        end
    case 'cartesian'
        throw('Not implemented')
        %         for coil = 1:CoilElements
        %             for roi = 1:NumOfRois
        %                 fprintf('Coil: %d/%d, ROI: %d/%d\n',coil,CoilElements,roi,NumOfRois)
        %                 image = imresize(tissue{roi,2}.*Sens_reco(:,:,coil),[Nsamples Nsamples],'Method','nearest');
        %                 Data_out = fftshift(fft2(image/nsamp));
        %                 Data_out = repmat(Data_out,1,ceil(echoes/Nsamples));
        %                 Data_out = Data_out(:,1:echoes);
        %                 R1 = tissue{roi,1};
        %                 SI = krho*sin(FA) * (1-exp(-R1*TR)) ./ (1-cos(FA)*exp(-R1*TR));
        %                 SI = repmat(SI,size(Data_out,1),1);
        %                 Data_out = Data_out.*SI;
        %                 EchoSignals(:,:,coil) = EchoSignals(:,:,coil) + Data_out;
        %             end
        %         end
end

% merge the four images into one
imgs = squeeze(sum(permute(imgs,[1 2 4 3]).*conj(Sensitivities),3));
if real_data
    % recalculate the trajectory from samples to radians [-pi:pi]*im_ovs
    k_traj(1,:) = k_traj(1,:)/max(k_traj(1,:))*pi*im_ovs;
    k_traj(2,:) = k_traj(2,:)/max(k_traj(2,:))*pi*im_ovs;
else
    k_traj = 0;
end

aif = subSample(aif, rpf);

end