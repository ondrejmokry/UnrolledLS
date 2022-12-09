clear
clc

if isfile('../../custom_config_simulator.json')
    fid = fopen('../../custom_config_simulator.json'); % Opening the file
else
    fid = fopen('../../default_config_simulator.json'); % Opening the file
end

raw = fread(fid,inf); % Reading the contents
str = char(raw'); % Transformation

hyper_parameters = jsondecode(str);
df = hyper_parameters.datafold;
factor = hyper_parameters.factor;
n_test_exp = 7;

FA = linspace(3,35,n_test_exp);

data1 = cell(1,n_test_exp);
fprintf('Generating calibration scans\n')
for i = 1:length(FA)
    [imgs,~,~,~,~,~,~] = simulate_acq_16_DCE_analysis(FA(i),0,1);
    tmp = single(abs(imgs(:,:,1:20)));
    data1{i} = Mat2DataCells(single(tmp))';
end

FA = [FA, hyper_parameters.FA];
[imgs, k_traj, EchoSignals, perfparams_out, Ts, aif, Sensitivities] = simulate_acq_16_DCE_analysis(hyper_parameters.FA,1,0);

EchoSignals = single(EchoSignals);
k_traj = single(k_traj);
imgs = single(imgs);
Sensitivities = single(Sensitivities);

data2 = Mat2DataCells(abs(imgs))';

date = datestr(clock,30);

outname = fullfile(df, 'SyntheticEchoes');
outname_aif = fullfile(df, 'aif');
outname = [outname, '_', date];
outname_aif = [outname_aif, '_', date];

rpf = hyper_parameters.rpf;
SD = hyper_parameters.SD_noise;

info.experiment = 'data_sim';
info.section = 1;
info.sections = 1;
info.acq.TR = repmat(hyper_parameters.TR*1e3,length(FA),1)';
info.acq.FA = FA;
info.acq.TE = ones(size(FA));
info.acq.kM0Range = [1,length(data1)];
info.acq.Ts = Ts;

experiment = info.experiment;

%% save echo signals
save(outname,...
    'EchoSignals',...
    'factor',...
    'imgs',...
    'k_traj',...
    'rpf',...
    'Sensitivities',...
    'SD',...
    'data1',...
    'info',...
    'perfparams_out',...
    '-v7.3')

%% save AIF
save(outname_aif,...
    'aif',...
    'experiment')