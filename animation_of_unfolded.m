clear
clc
close all

savefigs = false;
subframe = 4;

% set interpreter to tex
set(groot, 'defaultAxesTickLabelInterpreter', 'none');
set(groot, 'defaultColorbarTickLabelInterpreter', 'none');
set(groot, 'defaultLegendInterpreter', 'none');
set(groot, 'defaultTextInterpreter', 'none');

% read location of the data
fileID = fopen('../datafold.txt', 'r');
df = fscanf(fileID, '%s');
fclose(fileID);

% get files
[files, path] = uigetfile(df, 'Select file(s)', 'MultiSelect', 'on');
if ~path
    return
end
if iscell(files)
    nofiles = length(files);
else
    nofiles = 1;
    files = {files};
end

% load data
if nofiles == 1
    fprintf('Loading file...\n')
else
    fprintf('Loading files...\n')
end
D = struct(load([path, files{1}]));
for n = 1:nofiles
    D(n) = load([path, files{n}]);
end

% number of frames to plot
Nf = Inf;
for n = 1:nofiles
    Nf = min(Nf, length(D(n).data2));
end

%% animation
% initialize the figure
fig = figure('Position', [300 300 300*nofiles 500]);
tiles = tiledlayout(1, nofiles, 'Padding', 'none', 'TileSpacing', 'tight');
set(fig,'units','centimeters')
set(fig,'paperunits','centimeters')
set(fig,'papersize',fig.Position(3:4))

% compute future colorbar limits
fprintf('Computing limits...\n')
% cmin = Inf;
cmax = -Inf;
for n = 1:nofiles
    for f = 1:subframe:Nf
        % cmin = min(cmin, min(D(n).data2{f}(:)));
        cmax = max(cmax, max(D(n).data2{f}(:)));
    end
end
cmin = 0;

% prepare filenames
filenames = cell(nofiles,1);
fileids = zeros(nofiles,1);
for n = 1:nofiles
    switch files{n}(33:end-4)
        case 'gtc'
            filenames{n} = 'ground truth';
            fileids(n) = 1;
        case 'CP'
            filenames{n} = 'classical L+S';
            fileids(n) = 3;
        case 'CP2'
            filenames{n} = 'classical L+S repeated';
            fileids(n) = 4;
        case 'simple'
            filenames{n} = 'inverse NUFT';
            fileids(n) = 2;
        case 'unfolded'
            filenames{n} = 'unfolded';
            fileids(n) = 5;
        case 'reunfolded'
            filenames{n} = 'unfolded repeated';
            fileids(n) = 6;
    end
end

% reorder
[~, I] = sort(fileids);
files = files(I);
filenames = filenames(I);
D = D(I);

% initialize all the plots
img = gobjects(nofiles, 1);
for n = 1:nofiles
    nexttile
    img(n) = imagesc(D(n).data2{1}, [cmin, cmax]);
    title(filenames{n})
    axis square
    axis off
    if n == nofiles
        colorbar
    end
end

% initialize title
ttl = title(tiles, 'frame 1');

% plot all the frames
fprintf('Aniamtion running...\n')
figcounter = 0;
for f = 1:subframe:Nf
    for n = 1:nofiles
        img(n).CData = D(n).data2{f};
    end
    ttl.String = sprintf('frame %d\n\n', f);
    drawnow
    
    % save
    if savefigs
        pause(1)
        % print(fig,['../animations/unfolded-', num2str(figcounter)],'-dpdf','-bestfit') %#ok<*UNRCH>
        print(fig,['D:\OneDrive - Vysoké učení technické v Brně\SPLab\2022_AKTION\animations\unfolded-', num2str(figcounter)],'-dpdf','-bestfit') %#ok<*UNRCH>       
    end
    figcounter = figcounter + 1;
end

%% perfusion curves
% initialize the figure
fprintf('Ploting perfusion curves...\n')
fig = figure('Position', [100 100 300*(nofiles+2) 800]);
set(fig,'units','centimeters')
set(fig,'paperunits','centimeters')
set(fig,'papersize',fig.Position(3:4))
tiledlayout(2, nofiles, 'Padding', 'none', 'TileSpacing', 'tight');

% choose the frame to plot
plotframe = round(Nf/2);

% plot example frame
s = gobjects(nofiles, 1);
for n = 1:nofiles
    s(n) = nexttile;
    imagesc(D(n).data2{plotframe}, [cmin, cmax])
    title(filenames{n})
    axis square
    axis off
    if n == nofiles
        colorbar
    end
end

% prepare tiles for the curves
t = nexttile([1 nofiles]);
% title(t, 'perfusion curve (signal intensity)')
xlim(t, [1, Nf])
ylim(t, [cmin, cmax])
box(t, 'on')
grid(t, 'on')
pl = line(t, plotframe*[1, 1], [cmin, cmax],...
            'color', [.5 .5 .5],...
            'linewidth', 1.5,...
            'displayname', 'plotted frame');
xlabel(t, 'frame')
ylabel(t, 'signal intensity')

% plot the curves
X = size(D(1).data2{1}, 2);
Y = size(D(1).data2{1}, 1);
x = NaN;
y = NaN;
while 1
    try
        [newx, newy] = ginput(1);
        if gca == t
            plotframe = round(newx);
            plotframe = max(1, min(plotframe, Nf));
        else
            x = round(newx);
            y = round(newy);
            x = max(1, min(x, X));
            y = max(1, min(y, Y));
        end
    catch
        fprintf('Mischief managed!\n')
        break
    end
    for n = 1:nofiles
        hold(s(n), 'on')
        imagesc(s(n), D(n).data2{plotframe}, [cmin, cmax])
        if ~isnan(x)
            scatter(s(n), x, y, 'r')
        end
        hold(s(n), 'off')
    end

    % the resulting curve
    if ~isnan(x)
        cla(t)
        hold(t, 'on')
        for n = 1:nofiles
            curve = NaN(Nf, 1);
            for f = 1:Nf
                curve(f) = D(n).data2{f}(y, x);
            end
            plot(t, curve, 'displayname', filenames{n}, 'linewidth', 1)
        end
        line(t, plotframe*[1, 1], [cmin, cmax],...
            'color', [.5 .5 .5],...
            'linewidth', 1.5,...
            'displayname', 'plotted frame');
        hold(t, 'off')
        legend(t, 'location', 'northeast')
        xlabel(t, 'frame')
        ylabel(t, 'signal intensity')
        % title(t, sprintf('perfusion curve (signal intensity) at [%d, %d]', x, y))
    else
        pl.XData = plotframe*[1, 1];
    end
end