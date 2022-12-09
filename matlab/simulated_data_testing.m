% the script compares the data simulated using linear interpolation, nufft
% and dtft
%
% Date: 13/01/2021
% By Ondrej Mokry
% Brno University of Technology
% Contact: ondrej.mokry@mensa.cz

clear
close all
clc
% rng(0)

fileID = fopen('../datafold.txt','r');
df = fscanf(fileID,'%s');
fclose(fileID);

%% load the data
datafold = [df, '/simulation/'];
LI = load([datafold, 'SyntheticEchoes_linear_interp.mat'],'EchoSignals');
NU = load([datafold, 'SyntheticEchoes_nufft.mat'],'EchoSignals');
DT = load([datafold, 'SyntheticEchoes_dtft.mat'],'EchoSignals');
n  = size(LI.EchoSignals,1);

%% choose the radial and coil element to plot
% radial = randi(size(LI.EchoSignals,2));
radial = randi(168);
coil   = randi(size(LI.EchoSignals,3));

%% plot the spectrum along the chosen radial
plotdata = [LI.EchoSignals(:,radial,coil),...
    NU.EchoSignals(:,radial,coil),...
    DT.EchoSignals(:,radial,coil) ];

figure

% real part of selected coefficients
subplot(2,2,1)
plot(real(plotdata))
legend('interpolation','nufft per radial','dtft')
title('real part')

% imaginary part of selected coefficients
subplot(2,2,3)
plot(imag(plotdata))
legend('interpolation','nufft per radial','dtft')
title('imaginary part')

% differences
subplot(2,2,[2 4])
semilogy(abs(plotdata(:,1:2) - repmat(plotdata(:,3),1,2)))
legend('interpolation','nufft per radial')
title('absolute difference to by hand')

sgtitle(['radial number ',num2str(radial), ', coil element number ',num2str(coil)])

%% command window output
fprintf('Maximum absolute differences of the coefficients\n')
fprintf('  (1) linear interpolation\n')
fprintf('  (2) NUFFT separately per each radial\n')
fprintf('  (3) dtft by Fessler\n\n')
dists = NaN(size(plotdata,2));
for i = 1:size(plotdata,2)
    for j = 1:size(plotdata,2)
        dists(i,j) = max(abs(plotdata(:,i)-plotdata(:,j)));
    end
end
disp(dists)