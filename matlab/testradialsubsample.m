clear
clc
close all

TR = 7.5e-3; % [s] odvozeno z realnych dat
flength = 1; % [s] ocekavana delka casoveho okna
rpf = floor(flength/TR); % pocet radial na snimek

% redukce rpf na mocninu 2
rpf = 2^(nextpow2(rpf) - 1);

% vysledna delka casoveho okna
flength = TR * rpf; % [s]

%% plne vzorkovani
GoldenAngle = 2*pi /(1+sqrt(5)); % [rad]
Angles = (0:rpf-1) * GoldenAngle; % [rad]

% radialy k-prostoru
kx = [-1; 1] * cos(Angles);
ky = [-1; 1] * sin(Angles);

% vykresleni
figure
plot(kx, ky)
xlabel('Real part')
ylabel('Imaginary part')
axis square
title('Fully sampled frame')

%% povdzorkovani
figure
tiledlayout('flow')
for i = 2:log2(rpf)-2
    
    factor = 2^i;
    subkx = kx(:, 1:factor:end);
    subky = ky(:, 1:factor:end);
    
    nexttile
    plot(subkx, subky)
    xlabel('Real part')
    ylabel('Imaginary part')
    axis square
    title(sprintf('Subsampling factor: %d (%d radials)', factor, size(subkx, 2)))

end