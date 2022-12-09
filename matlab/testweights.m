% the script illustrates the computation of density compensating weights
% via the (updated) function DoCalcDCF
%
% Date: 18/02/2021
% By Ondrej Mokry
% Brno University of Technology
% Contact: ondrej.mokry@mensa.cz

clear
clc
close all
addpath('reconstruction')
%#ok<*UNRCH>

% settings
Nsamples  = 17;    % number of acquired samples per each radial line
rads      = 5;     % number of radial lines
frames    = 4;     % number of frames
X         = 128;   % image width
Y         = 128;   % image height
colors    = parula(rads); % color order for the plots
ordercor  = true; % if true, the modified DoCalcDCF is used
bordercor = false; % if true, the first and the last value per each radial
                   % is recomputed (instead of using 0)

% acquired angles
golden = 2*pi /(1+sqrt(5));
angles = (0:frames*rads-1)*golden;

% these are the acquired locations in the k-space
if mod(Nsamples,2)
    % the symmetric case
    kx = ( -floor(Nsamples/2):floor(Nsamples/2) )' * cos(angles);
    ky = ( -floor(Nsamples/2):floor(Nsamples/2) )' * sin(angles);
else
    % the not so symmetric case
    kx = ( (-Nsamples/2):(Nsamples/2)-1 )' * cos(angles);
    ky = ( (-Nsamples/2):(Nsamples/2)-1 )' * sin(angles);
end
kx = reshape(kx, [], frames)/X;
ky = reshape(ky, [], frames)/Y;

% density compensation
w = zeros(rads*Nsamples, frames);
figure
for f = 1:frames
    if ordercor
        w(:,f) = DoCalcDCF(kx(:,f), ky(:,f))*X*Y;
    else
        w(:,f) = oldDoCalcDCF(kx(:,f), ky(:,f))*X*Y;
    end
    
    % plot the Voronoi diagram
    subplot(2,frames,f)
    colororder([[0.5, 0.5, 0.5]; colors])
    voronoi(kx(:,f), ky(:,f))
    title(['frame ', num2str(f)])
    axis square
    
    % indicate the radials
    hold on
    for r = 1:rads       
        scatter(kx((r-1)*Nsamples + 1 : r*Nsamples, f),...
                ky((r-1)*Nsamples + 1 : r*Nsamples, f))        
    end
    
    % eliminate the zeros on the edge of the Voronoi diagram
    if bordercor
        for r = 1:rads
            % the first value per each radial
            w(1+(r-1)*Nsamples,f) = 2*w(2+(r-1)*Nsamples,f) - w(3+(r-1)*Nsamples,f);

            % the last value per each radial
            w(r*Nsamples,f) = 2*w(r*Nsamples-1,f) - w(r*Nsamples-2,f);
        end
    end
    
    % plot the weights
    subplot(2,frames,frames + f)
    colororder(colors)
    plot(reshape(w(:,f),[],rads))
end