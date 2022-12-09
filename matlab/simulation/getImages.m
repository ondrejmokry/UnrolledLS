function imgs = getImages(time,factor)
% getImages computes the simulated images in given time instants
%
% Date: 03/02/2020
% By Ondrej Mokry
% Brno University of Technology
% Contact: ondrej.mokry@mensa.cz

fileID = fopen('../../datafold.txt','r');
df = fscanf(fileID,'%s');
fclose(fileID);

%% input file name
inpname = [df, '/simulation/mapy.mat_a_tis_170810.mat'];

% resize factor
ResizeFactor = 0.8;

%% global tissue parameters
r1  = 3.2; % r1 relaxivity of tissue and blood [l/mmol/s]
R10 = 1;   % 1/s

%% acquisition parameters
TR = 0.015;     % TR [s] (time interval between radial acquisition)
FA = 20*pi/180; % flip angle of excitation RF pulses [rad]
delay = 30;     % bolus arrival time [s]
krho = 0.01;    % multiplication factor accounting for spin density, gains,... here set arbitrarily, don't change (it would change the SNR which is now tuned to agree with real data)

%% time axis in [min]
% temporal sampling period for generation of the curves [s], now equal to
% TR, shouldn't be changed
Ts = TR;

% number of frames (number of temporal samples) of the concentration curves
N = 30000;

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
load(inpname,'tissue','data2')
NumOfRois = size(tissue,1);

%% modify input geometry
xsize   = size(data2{1},2);
ysize   = size(data2{1},1);
for roi = 1:NumOfRois % for each ROi
    map = full(tissue{roi,2});
    if ResizeFactor == 1
        tissue{roi,2} = logical(map); %#ok<*AGROW>
    else
        newmap = imresize(map,ResizeFactor,'nearest');
        map = zeros(size(map));
        yi  = int32(round(ysize-size(newmap,1))/2);
        xi  = int32(round(xsize-size(newmap,2))/2);
        
        % newmap is placed in the middle with respect to the original map
        map(yi:yi+size(newmap,1)-1, xi:xi+size(newmap,2)-1) = newmap;
        tissue{roi,2} = logical(map);
    end    
end

%% generate time curves
echos = floor(time/Ts);

% time curve for each ROI (and reference perfusion-parameter maps)
Fp = zeros(ysize,xsize);
E  = zeros(ysize,xsize);
ve = zeros(ysize,xsize);
Tc = zeros(ysize,xsize);
for roi = 1:NumOfRois
    parametry = [tissue{roi,1}(1); tissue{roi,1}(2); tissue{roi,1}(3); tissue{roi,1}(4)];
    Fp(tissue{roi,2}) = parametry(1);
    E(tissue{roi,2})  = parametry(2);
    ve(tissue{roi,2}) = parametry(3);
    Tc(tissue{roi,2}) = parametry(4);
    TRF  = DCATH_v5_complete(parametry,t);
    conc = Ts/60 * real(ifft(   fft(TRF,NN) .* fft(aif,NN)   ));
    conc = conc(1:int32(N));
    R1   = conc*r1+R10; % R1(t) curve

    tissue{roi,4} = tissue{roi,1};
    tissue{roi,1} = R1;
end

% initialize images
outy = round(ysize*factor);
outx = round(xsize*factor);
imgs = zeros(outy,outx,length(echos));
for echo = 1:length(echos)
    
    % compute image for the current echo
    img = zeros(ysize,xsize);
    for roi = 1:NumOfRois  % for each ROi
        % compute SI (signal intensity) of the given ROI and time instant
        R1 = tissue{roi,1}(echos(echo));
        SI = krho*sin(FA) * (1-exp(-R1*TR)) ./ (1-cos(FA)*exp(-R1*TR));
        img(tissue{roi,2}) = SI;
    end
    imgs(:,:,echo) = imresize(img,[outy outx]);
    
end

end