clear
clc
close all
rng(0)

%% load the data
fileID = fopen('../../datafold.txt','r');
datafold = fscanf(fileID,'%s');
fclose(fileID);
[file, path] = uigetfile([datafold, '/reconstruction/*.mat'],'Select a file');
if file == 0
    return
end
load([path, file])
[Y, X, Nf] = size(simple);

if strcmp(file(1:6),'INUFFT')
    algo = 2;
else
    algo = 1;
end

if strcmp(file(1:4),'real')
    variant = 1;
else
    variant = 2;
end

% choose the frame to plot
plotframe = randi(Nf);

%% compute and plot the shift maps
if algo == 1 && variant > 1  
    % compute
    solutiondif = abs(solution)-abs(gtc);
    simpledif   = abs(simple)-abs(gtc);

    % plot
    figure
    subplot(2,3,1)
    imagesc(mean(solutiondif,3))
    colorbar
    axis square
    title('L+S minus scaled ground truth, mean')

    subplot(2,3,2)
    imagesc(median(solutiondif,3))
    colorbar
    axis square
    title('L+S minus scaled ground truth, median')

    subplot(2,3,3)
    imagesc(mode(solutiondif,3))
    colorbar
    axis square
    title('L+S minus scaled ground truth, mode')

    subplot(2,3,4)
    imagesc(mean(simpledif,3))
    colorbar
    axis square
    title('iNUFFT minus scaled ground truth, mean')

    subplot(2,3,5)
    imagesc(median(simpledif,3))
    colorbar
    axis square
    title('iNUFFT minus scaled ground truth, median')

    subplot(2,3,6)
    imagesc(mode(simpledif,3))
    colorbar
    axis square
    title('iNUFFT minus scaled ground truth, mode')
end

%% plot reconstructed frame
figure
if variant == 1
    A = 2; B = 1;    
elseif algo == 1
    A = 3; B = 3;
else
    A = 2; B = 3;
end
s = subplot(A,B,1);
if algo == 1
    imagesc(abs(solution(:,:,plotframe)))
else
    imagesc(abs(simple(:,:,plotframe)))
end
hold on
colorbar
axis square
title({sprintf('reconstructed frame %d of %d',plotframe,Nf),'click to show a perfusion curve, close the figure to end'})

%% plot ground truth frame
if variant > 1
    subplot(A,B,2)
    imagesc(abs(gt(:,:,plotframe)))
    hold on
    colorbar
    axis square
    title(sprintf('ground truth frame %d of %d',plotframe,Nf))
    
    subplot(A,B,3)
    imagesc(abs(gtc(:,:,plotframe)))
    hold on
    colorbar
    axis square
    title({sprintf('ground truth frame %d of %d',plotframe,Nf),...
        'multiplied with the sensitivity norms'})
end

%% plot perfusion curves
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
    if algo == 1
        imagesc(s,abs(solution(:,:,plotframe)))
    else
        imagesc(s,abs(simple(:,:,plotframe)))
    end
    scatter(s,x,y,'r')

    if variant == 1 % real data
        
        subplot(A,B,2)
        yyaxis left
        plot(timeaxis,squeeze(abs(solution(y,x,:))))
        yyaxis right
        plot(timeaxis,squeeze(abs(simple(y,x,:))))
        legend('L+S','inverse NUFFT')
        xlabel('time / s')
        title(sprintf('perfusion curve at [%d,%d]',x,y))
        ax = gca;
        ax.XGrid = 'on';
        ax.XMinorGrid = 'on';
        xlim([min(timeaxis), max(timeaxis)])
        
    else % simulated data
        
        % the resulting curve
        subplot(A,B,4:6)
        yyaxis left
        if algo == 1
            plot(timeaxis,squeeze(abs(solution(y,x,:))),...
                 timeaxis,squeeze(abs(simple(y,x,:))),':',...
                 timeaxis,squeeze(abs(gtc(y,x,:))),'--')
        else
            plot(timeaxis,squeeze(abs(simple(y,x,:))),':',...
                 timeaxis,squeeze(abs(gtc(y,x,:))),'--')
        end
        yyaxis right
        plot(timeaxis,squeeze(abs(gt(y,x,:))))
        if algo == 1
            legend('L+S','inverse NUFFT','scaled ground truth','ground truth')
        else
            legend('inverse NUFFT','scaled ground truth','ground truth')
        end
        xlabel('time / s')
        title(sprintf('perfusion curve at [%d,%d]',x,y))
        ax = gca;
        ax.XGrid = 'on';
        ax.XMinorGrid = 'on';
        xlim([min(timeaxis), max(timeaxis)])
        
        % L and S components
        if algo == 1
            subplot(A,B,7:9)
            plot(timeaxis,squeeze(abs(x_new{1}(y,x,:))))
            hold on
            plot(timeaxis,squeeze(abs(x_new{2}(y,x,:))))
            hold off
            xlabel('time / s')
            title(sprintf('L and S components at [%d,%d]',x,y))
            ax = gca;
            ax.XGrid = 'on';
            ax.XMinorGrid = 'on';
            xlim([min(timeaxis), max(timeaxis)])
        end
    end
end