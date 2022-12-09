clear
clc
close all

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
[file, path] = uigetfile(df, 'Select file(s)', 'MultiSelect', 'on');
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
    fprintf('Loading file...\n')
    D = load([path, file]);
else
    fprintf('Loading files...\n')
    D = struct(load([path, file{1}]));
    for n = 1:nofiles
        D(n) = load([path, file{n}]);
    end
end

% number of frames to plot
Nf = Inf;
for n = 1:nofiles
    Nf = min(Nf, length(D(n).data2));
end

%% animation
% initialize the figure
figure('Position', [300 300 400*nofiles 600]);
tiles = tiledlayout(1, nofiles, 'Padding', 'none', 'TileSpacing', 'tight');

% compute future colorbar limits
fprintf('Computing limits...\n')
cmin = Inf;
cmax = -Inf;
for n = 1:nofiles
    for f = 1:Nf
        cmin = min(cmin, min(D(n).data2{f}(:)));
        cmax = max(cmax, max(D(n).data2{f}(:)));
    end
end

% initialize all the plots
img = gobjects(nofiles, 1);
for n = 1:nofiles
    nexttile
    img(n) = imagesc(D(n).data2{1}, [cmin, cmax]);
    if nofiles > 1
        title(file{n})
    else
        title(file)
    end
    axis square
    colorbar
end 
    
% initialize title
ttl = title(tiles, 'frame 1');

% plot all the frames
fprintf('Aniamtion running...\n')
for f = 1:Nf
    for n = 1:nofiles
        img(n).CData = D(n).data2{f};
    end
    ttl.String = sprintf('frame %d\n\n', f);
    drawnow
end

%% perfusion curves
% initialize the figure
fprintf('Ploting perfusion curves...\n')
figure('Position', [100 100 300*(nofiles+2) 800]);
tiledlayout(2, nofiles, 'Padding', 'none', 'TileSpacing', 'tight');

% choose the frame to plot
plotframe = round(Nf/2);

% plot example frame
s = gobjects(nofiles, 1);
for n = 1:nofiles
    s(n) = nexttile;
    imagesc(D(n).data2{plotframe}, [cmin, cmax])
    if nofiles > 1
        title(file{n})
    else
        title(file)
    end
    axis square
    colorbar
end

% prepare tiles for the curves
t = nexttile([1 nofiles]);
title(t, 'perfusion curve')
xlim(t, [1, Nf])
ylim(t, [cmin, cmax])
box(t, 'on')
grid(t, 'on')
pl = line(t, plotframe*[1, 1], [cmin, cmax], 'color', 'r', 'displayname', 'plotted frame');

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
            if nofiles > 1
                plot(t, curve, 'displayname', file{n})
            else
                plot(t, curve, 'displayname', file)
            end
        end
        line(t, plotframe*[1, 1], [cmin, cmax], 'color', 'r', 'displayname', 'plotted frame');
        hold(t, 'off')
        legend(t, 'location', 'northeast')
        title(t, sprintf('perfusion curve at [%d, %d]', x, y))
    else
        pl.XData = plotframe*[1, 1];
    end
end