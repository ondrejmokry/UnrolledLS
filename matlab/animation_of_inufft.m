clear
clc
close all

% save files?
savefigs = false;

% set interpreter to tex
set(groot, 'defaultAxesTickLabelInterpreter','tex');
set(groot, 'defaultColorbarTickLabelInterpreter','tex');
set(groot, 'defaultLegendInterpreter','tex');
set(groot, 'defaultTextInterpreter','tex');

% read location of the data
fileID = fopen('../datafold.txt','r');
df = fscanf(fileID,'%s');
fclose(fileID);

% get files
[file, path] = uigetfile([df, '/reconstruction/LS_*.mat'],'Select file');
if ~path
    return
end

% load data
D = load([path,file]);

% number of frames to plot
Nf = D.Nf;

%% animation
% initialize the figure
fig = figure('Position',[100 100 700 300]);
set(fig,'units','centimeters')
set(fig,'paperunits','centimeters')
set(fig,'papersize',fig.Position(3:4))
tiles = tiledlayout(1, 2, 'Padding', 'none', 'TileSpacing', 'loose');

% compute future colorbar limits
cmin = min(D.solution(:));
cmax = max(D.solution(:));

% initialize all the plots
nexttile
img_gtc = imagesc(D.gtc(:,:,1), [cmin, cmax]);
title('ground truth')
axis square
colorbar

nexttile
img_simple = imagesc(D.simple(:,:,1), [cmin, cmax]);
title('not regularized')
axis square
colorbar
    
% initialize title
ttl = title(tiles,'frame 1');

% plot all the frames
figcounter = 0;
for f = 1:Nf
    img_gtc.CData    = D.gtc(:,:,f);
    img_simple.CData = D.simple(:,:,f);
    ttl.String  = sprintf('frame %d',f);
    drawnow
    
    % save
    if savefigs
        print(fig,['../animations/INUFFT-', num2str(figcounter)],'-dpdf','-bestfit') %#ok<*UNRCH>
    end
    figcounter = figcounter + 1;
end