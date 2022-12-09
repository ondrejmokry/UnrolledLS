% Use this single script to generate synthetic DCE-MRI data

clear
clc
close all
rng(0)

customname = '_32';

% add the package
fileID = fopen('../../packagefold.txt','r');
packagefold = fscanf(fileID,'%s');
fclose(fileID);
addpath(genpath([packagefold, '/ESPIRiT']));

fileID = fopen('../../datafold.txt','r');
df = fscanf(fileID,'%s');
fclose(fileID);

%% input file name
% shuldn't be changed; includes ROIs and the corresponding perf. parameters
inpname = [df, '/simulation/mapy.mat_a_tis_170810.mat'];

%% coil sensitivities
% taken from the same recording, preconstrast frames, heavily smoothed
sensitivities = [df, '/simulation/sensitivities.mat'];

% resize factor
ResizeFactor = 0.8;

%% output file name and path
[pathstr, name, ext] = fileparts(inpname);
outname = fullfile(pathstr, 'SyntheticEchoes');

%% some global parameters
% plotting
showres = false; % if true, figures are shown during the calculation (slows down)
drawnow

% sampling type
cartesian = false;

% k-space radial interpolation options
sampling = 2;
switch sampling
    % (1) linear_interp ...... linear interpolations
    case 1, k_space_sampling = 'linear_interp';
    % (2) nufft .............. nufft for whole projection
    case 2, k_space_sampling = 'nufft';
    % (3) dtft ............... dtft
    case 3, k_space_sampling = 'dtft';
    % (4) nufft_per_coeff .... nufft for each position
    case 4, k_space_sampling = 'nufft_per_coeff';
end
outname = [outname, '_', k_space_sampling, customname];

% Create acquisition scheme:
% list of acquisition steps, for each TR (i.e. for each echo, or set of
% echoes for array coils) the "acquired" echo signal is specified as one
% entry in the following two vectors:
%   Angles - angle of the acquisition trajectory line [rad]
%   PhEncShifts - shift in the phase encoding direction [samples in the k-space]

if cartesian
    % Cartesian sampling (for Michal Bartos, or for reference), settings used in our oncomice recordings:
    PhEncSteps  = 96; 
    Repetitions = 400;
    Angles = zeros(1,PhEncSteps*Repetitions); % angles - for Cartesian sampling always zero
    PhEncShifts = (-PhEncSteps/2):(PhEncSteps/2-1); % phase encoding step increments
    PhEncShifts = repmat(PhEncShifts,1,Repetitions);
else
    % Radial 2D golden angle sampling:
    Projections = 30000; %#ok<*UNRCH> % affects acquisition length (=Projections * TR) should be at least cca 10 min (standard duration: 10 - 15 min)
    GoldenAngle = 2*pi /(1+sqrt(5)); % [rad]
    Angles = (0:(Projections-1)) * GoldenAngle; % angles using golden angle increment [rad]
    PhEncShifts = zeros(1,Projections); % phase encoding shifts - for radial sampling always zero
end

%% global tissue parameters
r1  = 3.2; % r1 relaxivity of tissue and blood [l/mmol/s]
R10 = 1;   % [1/s]

%% acquisition parameters
TR = 0.015;     % TR [s] (time interval between radial acquisition)
FA = 20*pi/180; % flip angle of excitation RF pulses [rad]
Nsamples = 32;  % number of samples of each echo (sampling rate is fixed, given by the FOV)
delay = 30;     % bolus arrival time [s]

krho = 0.01; % multiplication factor accounting for spin density, gains,... here set arbitrarily, don't change (it would change the SNR which is now tuned to agree with real data)
% SD = 0.14; % standard deviation of noise
SD = 0;

%% time axis [min]
% temporal sampling period for generation of the curves [s], now equal to
% TR, shouldn't be changed
Ts = TR;

% number of frames (number of temporal samples) of the concentration curves
N  = length(Angles);

% time axis [min]
t  = (0:(N-1))*Ts/60;

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
    
    % show the ROIs
    if showres
        figure(1)
        subplot(6,7,roi)
        imagesc(testmap,'AlphaData',0.5)
        hold on
        imagesc(map,'AlphaData',0.5)
        axis off
        sgtitle('ROIs')
    end
    
end
% show the (logical) sum of the masks
if showres
    figure(2)
    imagesc(testmap)
    title('testmap')
end

%% generate time curves
% time curve for each ROI (and reference perfusion-parameter maps)
Fp = zeros(ysize,xsize);
E  = zeros(ysize,xsize);
ve = zeros(ysize,xsize);
Tc = zeros(ysize,xsize);
if showres
    figure(3)
    subplot(411)
    h(1) = plot(t,zeros(size(t))); xlabel('t [min]'); title('AIF [mmol]')
    subplot(412)
    h(2) = plot(t,zeros(size(t))); xlabel('t [min]'); title('F*TRF')
    subplot(413)
    h(3) = plot(t,zeros(size(t))); xlabel('t [min]'); title('C(t) [mmol]')
    subplot(414)
    h(4) = plot(t,zeros(size(t))); xlabel('t [min]'); title('R1(t) [1/s]')
end
for roi = 1:NumOfRois
    fprintf('ROI %2d / %d ', roi, NumOfRois)
    parametry = [tissue{roi,1}(1); tissue{roi,1}(2); tissue{roi,1}(3); tissue{roi,1}(4)];
    Fp(tissue{roi,2}) = parametry(1);
    E(tissue{roi,2})  = parametry(2);
    ve(tissue{roi,2}) = parametry(3);
    Tc(tissue{roi,2}) = parametry(4);
    TRF = DCATH_v5_complete(parametry,t);
    tic
    conc = Ts/60 * real(ifft(   fft(TRF,NN) .* fft(aif,NN)   ));
    conc = conc(1:int32(N));
    R1 = conc*r1+R10; % R1(t) curve
    toc

    tissue{roi,4} = tissue{roi,1};
    tissue{roi,1} = R1;
    if showres
        set(h(1),'YData',aif)
        set(h(2),'YData',TRF)
        set(h(3),'YData',conc)
        set(h(4),'YData',R1)
        pause(0.1)
    end
end
vp     = Tc.*Fp;
Ktrans = Fp .* E;
kep    = E .* Fp ./ ve;
PS     = -log(1-E) .* Fp * (1-0.7*0.4);

%% save reference parfusion-parameter maps
save([outname, '_ReferenceMaps'],'Fp','E','ve','Tc','vp','Ktrans','kep','PS')

%% generate echo signals
load(sensitivities);
CoilElements = size(Sensitivities,3);

if showres
    figure(4)
    subplot(121)
    imagesc(abs(Sensitivities(:,:,1)).*testmap)
    title('sensitivity times testmap')
    
    subplot(122)
    imagesc(abs(Sensitivities(:,:,1)))
    title('sensitivity')
end

echoes = length(Angles); % total number of echoes (TRs)
EchoSignals = zeros(Nsamples,echoes,CoilElements);

% k-space coordinates
kx = ( (-Nsamples/2):(Nsamples/2-1) )' * cos(Angles);
ky = ( (-Nsamples/2):(Nsamples/2-1) )' * sin(Angles) + repmat(PhEncShifts,[Nsamples, 1]);

% normalization to [-0.5, 0.5]
kx = kx/Nsamples;
ky = ky/Nsamples;

% second normalization to 0.5*[-Nsamples/xsize, Nsamples/ysize]
kx = kx*Nsamples/xsize;
ky = ky*Nsamples/ysize;

% saving the coordinates as complex numbers
k = complex(ky,kx); % ugly trick for NUFFT and dtft

if showres
    figure(5)
    subplot(131)
    h(1) = imagesc(zeros(ysize,xsize));
    colormap(gray)
    title('image times sensitivity')
    
    subplot(132)
    hold on
    h(2) = imagesc(zeros(ysize,xsize));
    h(3) = plot([0 0],[0 0]);
    title('k-space line location')
    xlim([-0.5 0.5]*xsize)
    ylim([-0.5 0.5]*ysize)
    
    subplot(133)
    h(4) = plot([0 0],[0 0]);
    title('echo signal')
end

for echo = 1:echoes % for each echo (TR)
    
    fprintf('echo %5d / %d ', echo, echoes)
    tic
    
    % create image of the given time instant 
    img = zeros(ysize,xsize);
    for roi = 1:NumOfRois  % for each ROi
        % compute SI (signal intensity) of the given ROI and time instant
        R1 = tissue{roi,1}(echo);
        SI = krho*sin(FA) * (1-exp(-R1*TR)) ./ (1-cos(FA)*exp(-R1*TR));
        img(tissue{roi,2}) = SI;
    end
    
    switch k_space_sampling
        case 'linear_interp'
            % nothing to precompute
        case 'nufft'
            % create NUFFT operator, same for all coils
            n_shift = [0 0];
            w  = 1;
            FT = NUFFT(k(:, echo), w, n_shift, [ysize xsize]);
        case 'dtft'
            n_shift = [ysize/2 xsize/2];
        case 'nufft_per_coeff'
            % create NUFFT operator for each coordinate
            n_shift = [0 0];
            w  = 1;
            FT = cell(1,Nsamples);
            for i = 1:Nsamples
                FT{1,i} = NUFFT(k(i, echo), w, n_shift, [ysize xsize]);
            end          
    end
    
    for CoilElement = 1:CoilElements % for each coil element
        
        % apply sensitivity of the given array element
        imgsens = img .* Sensitivities(:,:,CoilElement);
        if showres
            set(h(1),'CData',abs(imgsens))
        end
        
        % go to k-space
        Img = fftshift(fft2(fftshift(imgsens)));
        
        DCPhEnc   = size(Img,1)/2+1; % index of the 0-th k-space line in the phenc direction
        DCFreqEnc = size(Img,2)/2+1; % index of the 0-th k-space line in the freqenc direction
        
        switch k_space_sampling
            case 'linear_interp'           
                [X,Y] = meshgrid(1:size(Img,2),1:size(Img,1));
                X  = X - DCFreqEnc;
                Y  = Y - DCPhEnc;
                xi = ( (-Nsamples/2):(Nsamples/2-1) ) * cos(Angles(echo));
                yi = ( (-Nsamples/2):(Nsamples/2-1) ) * sin(Angles(echo)) + PhEncShifts(echo);
                EchoSignal = interp2(X,Y,Img,xi,yi,'linear');
            case 'nufft'
                EchoSignal = FT*imgsens;
                % scale such that it complies with the linear interpolation case
                EchoSignal = EchoSignal * sqrt(xsize*ysize);
            case 'dtft'
                EchoSignal = dtft(imgsens, 2*pi*[real(k(:, echo)) imag(k(:, echo))], n_shift, false);
            case 'nufft_per_coeff'
                EchoSignal = NaN(Nsamples,1);
                for i = 1:Nsamples
                    EchoSignal(i) = FT{i}*imgsens;
                end
                % scale such that it complies with the linear interpolation case
                EchoSignal = EchoSignal * sqrt(xsize*ysize);
        end
        
        % show k-space line location
        if showres
            set(h(2),'CData',log10(abs(Img)))
            set(h(2),'XData',(1:size(Img,2))-DCFreqEnc)
            set(h(2),'YData',(1:size(Img,1))-DCPhEnc)
            set(h(3),'XData',[xi(1) xi(end)])
            set(h(3),'YData',[yi(1) yi(end)])
        end
        
        % add zero-mean Gaussian noise
        noise = SD*randn(size(EchoSignal)) + 1i*SD*randn(size(EchoSignal));
        EchoSignal = EchoSignal + noise;
        if showres
            set(h(4),'XData',1:length(EchoSignal))
            set(h(4),'YData',abs(EchoSignal))
            pause(0.1)
        end
        
        % store
        EchoSignals(:,echo,CoilElement) = EchoSignal;
    end
    
    t = toc; % time per echo
    fprintf('Time per echo: %5.3f sec, remaining time: %6.3f hours\n', t, t*(echoes-echo)/3600)  
    clear imgsens img R1 SI Img X Y xi yi ZI EchoSignal noise
end

%% save echo signals
save(outname,'EchoSignals','Angles','PhEncShifts','TR','FA','delay','aif','N','r1')