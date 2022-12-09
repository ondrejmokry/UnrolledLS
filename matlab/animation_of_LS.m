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
[file, path] = uigetfile([df, '/reconstruction/LS_*.mat'],'Select file(s)','MultiSelect','on');
if ~path
    return
end
if iscell(file)
    nofiles = length(file);
else
    nofiles = 1;
end

% load data
if nofiles == 1
    D = load([path,file]);
else
    D = struct(load([path,file{1}]));
    for n = 1:nofiles
        D(n) = load([path,file{n}]);
    end
end

% number of frames to plot
Nf = Inf;
for n = 1:nofiles
    Nf = min(Nf,D(n).Nf);
end

%% animation
% initialize the figure
if nofiles > 1
    fig = figure('Position',[100 100 1600 300*nofiles]);
else
    fig = figure('Position',[100 100 1000 600]);
end
set(fig,'units','centimeters')
set(fig,'paperunits','centimeters')
set(fig,'papersize',fig.Position(3:4))
if nofiles > 1
    tiles = tiledlayout(nofiles, 5, 'Padding', 'none', 'TileSpacing', 'tight');
else
    tiles = tiledlayout(2, 1, 'Padding', 'compact', 'TileSpacing', 'loose');
end

% compute future colorbar limits
cmin = min(D(1).solution(:));
cmax = max(D(1).solution(:));

% initialize all the plots
img_gtc    = gobjects(nofiles,1);
img_simple = gobjects(nofiles,1);
img_LS     = gobjects(nofiles,1);
img_L      = gobjects(nofiles,1);
img_S      = gobjects(nofiles,1);
for n = 1:nofiles
    if nofiles == 1
        t1 = tiledlayout(tiles, 1, 2, 'Padding', 'loose', 'TileSpacing', 'tight');
    end
    if nofiles == 1
        nexttile(t1)
    else
        nexttile
    end
    img_gtc(n) = imagesc(D(n).gtc(:,:,1), [cmin, cmax]);
    title('ground truth')
    axis square
    colorbar

    if nofiles == 1
        nexttile(t1)
    else
        nexttile
    end
    img_simple(n) = imagesc(D(n).simple(:,:,1), [cmin, cmax]);
    title('not regularized')
    axis square
    colorbar

    if nofiles == 1
        t2 = tiledlayout(tiles, 1, 3, 'Padding', 'none', 'TileSpacing', 'tight');
        t2.Layout.Tile = 2;
    end
    if nofiles == 1
        nexttile(t2)
    else
        nexttile
    end
    img_LS(n) = imagesc(D(n).solution(:,:,1), [cmin, cmax]);
    title('L + S')
    axis square
    colorbar

    if nofiles == 1
        nexttile(t2)
    else
        nexttile
    end
    img_L(n) = imagesc(abs(D(n).x_new{1}(:,:,1)), [cmin, cmax]);
    title(sprintf('L (\\lambda = %s)',num2str(D(n).lambdaL)))
    axis square
    colorbar

    if nofiles == 1
        nexttile(t2)
    else
        nexttile
    end
    img_S(n) = imagesc(abs(D(n).x_new{2}(:,:,1)), [cmin, cmax]);
    title(sprintf('S (\\lambda = %s)',num2str(D(n).lambdaS)))
    axis square
    colorbar
end 
    
% initialize title
ttl = title(tiles,'frame 1');

% plot all the frames
figcounter = 0;
for f = 1:Nf
    for n = 1:nofiles
        img_gtc(n).CData    = D(n).gtc(:,:,f);
        img_simple(n).CData = D(n).simple(:,:,f);
        img_LS(n).CData     = D(n).solution(:,:,f);
        img_L(n).CData      = abs(D(n).x_new{1}(:,:,f));
        img_S(n).CData      = abs(D(n).x_new{2}(:,:,f));
    end
    ttl.String  = sprintf('frame %d',f);
    drawnow
    
    % save
    if savefigs
        print(fig,['../animations/LS-', num2str(figcounter)],'-dpdf','-bestfit') %#ok<*UNRCH>
    end
    figcounter = figcounter + 1;
end

%% perfusion curves
% initialize the figure
fig = figure('Position',[100 100 300*(nofiles+2) 800 ]);
set(fig,'units','centimeters')
set(fig,'paperunits','centimeters')
set(fig,'papersize',fig.Position(3:4))
tiledlayout(3, nofiles+2, 'Padding', 'none', 'TileSpacing', 'tight');

% choose the frame to plot
plotframe = round(Nf/2);

% plot ground truth
s = nexttile;
imagesc(D(1).gtc(:,:,plotframe), [cmin, cmax]);
title('ground truth')
axis square
colorbar

% plot non-regularized solution
nexttile
imagesc(D(1).simple(:,:,plotframe), [cmin, cmax]);
title('not regularized')
axis square
colorbar

% plot solutions
for n = 1:nofiles
    nexttile
    imagesc(D(n).solution(:,:,plotframe), [cmin, cmax]);
    title(sprintf('L (\\lambda = %s) + S (\\lambda = %s)',num2str(D(n).lambdaL),num2str(D(n).lambdaS)))
    axis square
    colorbar
end

% prepare tiles for the curves
t = nexttile([1 nofiles+2]);
u = nexttile([1 nofiles+2]);

% plot the curves
X = size(D(1).gtc,2);
Y = size(D(1).gtc,1);
while 1

    try
        [x, y] = ginput(1);
        x = round(x);
        y = round(y);
        x = max(1,min(x,X));
        y = max(1,min(y,Y));
    catch
        fprintf('Mischief managed!\n')
        break
    end
    hold(s,'on')
    imagesc(s,D(1).gtc(:,:,plotframe), [cmin, cmax]);
    scatter(s,x,y,'r')
    hold(s,'off')

    % the resulting curve
    cla(t)
    hold(t,'on')
    plot(t,D(1).timeaxis(1:Nf),squeeze(abs(D(1).gtc(y,x,1:Nf))),...
        'displayname','ground truth')
    plot(t,D(1).timeaxis(1:Nf),squeeze(abs(D(1).simple(y,x,1:Nf))),...
        'displayname','not regularized')
    for n = 1:nofiles
        plot(t,D(n).timeaxis(1:Nf),squeeze(abs(D(n).solution(y,x,1:Nf))),...
            'displayname',sprintf('L (\\lambda = %s) + S (\\lambda = %s)',num2str(D(n).lambdaL),num2str(D(n).lambdaS)))
    end
    hold(t,'off')
    legend(t,'location','northwest')
    title(t,sprintf('perfusion curve at [%d,%d]',x,y))
    xlim(t,D(1).timeaxis([1 Nf]))
    xlabel(t,'time (s)')
    % ylim(t,[cmin, cmax])
    box(t,'on')
    grid(t,'on')

    % L and S components
    C = colororder;
    cla(u)
    hold(u,'on')
    for n = 1:nofiles
        plot(u,D(n).timeaxis(1:Nf),squeeze(abs(D(n).x_new{1}(y,x,1:Nf))),...
            'color',C(2+n,:),...
            'linestyle','--',...
            'displayname',sprintf('L (\\lambda = %s)',num2str(D(n).lambdaL)))
        plot(u,D(n).timeaxis(1:Nf),squeeze(abs(D(n).x_new{2}(y,x,1:Nf))),...
            'color',C(2+n,:),...
            'linestyle',':',...
            'displayname',sprintf('S (\\lambda = %s)',num2str(D(n).lambdaS)))
    end
    hold(u,'off')
    legend(u,'location','northwest')
    xlabel(u,'time / s')
    title(u,sprintf('L and S components at [%d,%d]',x,y))
    xlim(u,D(1).timeaxis([1 Nf]))
    xlabel(u,'time (s)')
    % ylim(u,[cmin, cmax])
    box(u,'on')
    grid(u,'on')
    
    % save
    if savefigs
        print(fig,['../animations/LS-at-', num2str(x), '-', num2str(y)],'-dpdf','-bestfit') %#ok<*UNRCH>
    end
end