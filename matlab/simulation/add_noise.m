clear
clc
close all

rng(0)
subfold = 'train';
stds = [1e-4, 1e-3, 1e-2, 1e-1];

% budeme nacitat existujici simulovana data a pricitat k nim sum
fileID = fopen('../../datafold.txt','r');
df = fscanf(fileID,'%s');
fclose(fileID);

% vytvorime tabulku vsech souboru
files   = dir([df, '/', subfold, '/*.mat']);
nofiles = length(files); % pocet souboru

for f = 1:nofiles

    file = files(f).name;
    path = [files(f).folder, '/'];
    
    % vystup do prikazoveho okna
    fprintf('File: %s (%d/%d)',file,f,nofiles)
    
    if contains(file,'noise')
        fprintf(' skipped\n')
        continue
    else
        fprintf('\n')
    end
    
    % nacteni souboru
    data = load([path, file]);
    
    for SD = stds
        newdata = data;
        
        if ~contains(file, 'aif')
            % pricteni sumu
            noise = SD*randn(size(newdata.EchoSignals)) + 1i*SD*randn(size(newdata.EchoSignals));
            newdata.EchoSignals = newdata.EchoSignals + noise;
            newdata.SD = SD;
        end
        
        % ulozeni noveho souboru
        save([path, file(1:end-4), '_', strrep(num2str(SD,'%.4f'),'.',''), '.mat'],...
            '-struct',...
            'newdata',...
            '-v7.3')
    end

end
