% the script performs reconstruction from the synthetic DCE-MRI data from
% the simulator
%
% Date: 02/06/2021
% By Ondrej Mokry
% Brno University of Technology
% Contact: ondrej.mokry@mensa.cz

clear
clc
close all

% add the package
fileID = fopen('../../packagefold.txt','r');
packagefold = fscanf(fileID,'%s');
fclose(fileID);
addpath(genpath([packagefold, '/ESPIRiT']));

fileID = fopen('../../datafold.txt','r');
df = fscanf(fileID,'%s');
fclose(fileID);

rng(0)

%% load the data
datafold = [df, '/real/'];
datafile = '6_golden_angle_dynamika';
recofold = [df, '/reconstruction/'];

load([datafold, datafile])

[~,Ncoils] = size(signal);
Nsamples   = NumOfSampl;
N          = NumOfProj;
signal     = reshape(signal,[Nsamples N Ncoils]);
X          = 128;
Y          = 128;

% signal    the signal (array Nsamples x N x Ncoils)
% N         number of samples
% Ncoils    number of coil elements
% Nsamples  number of samples of each echo
% X, Y      size of the image

%% crop the beginning
cropsamples = 0; % crop interval as a number of echos
EchoSignals = signal(:,cropsamples+1:end,:);
kx          = kx(cropsamples*Nsamples+1:end);
ky          = ky(cropsamples*Nsamples+1:end);
N           = N - cropsamples;

%% settings
% radials per frame
rpf = 21;

% number of frames
Nf = 100;
Nf = min(Nf, floor(N/rpf));

% regularizers (parameters of the model)
lambdaS = 0.025;
lambdaL = 0.1;

% the Chambolle-Pock algorithm
theta      = 1;
tau        = 0.9;
sigma      = 1/(8*tau);
iterations = 20;

%% load the sensitivities
maps    = [];
weights = [];
load([datafold, 'citlivosti_kartezske'])
maps    = rot90(maps(:,end:-1:1,:),1);
weights = rot90(weights(:,end:-1:1),1);

%% use it to compute the sensitivities via ESPIRiT
ESP = ESPIRiT(maps,weights);

%% precompute the NUFFT operators
w   = zeros(Nsamples*rpf,Nf);
FFT = cell(Nf,1);
for f = 1:Nf
    
    fprintf('Defining the NUFFT for frame %d of %d.\n',f,Nf)
    
    kxf = kx((f-1)*rpf*Nsamples+1:f*rpf*Nsamples)'*(Nsamples/X);
    kyf = ky((f-1)*rpf*Nsamples+1:f*rpf*Nsamples)'*(Nsamples/Y);

    % compute the density compensation
    % the function DoCalcDCF has been edited such that the areas are not
    % normalized
    w(:,f) = DoCalcDCF(kxf, kyf);
    w(:,f) = w(:,f)/max(w(:,f));
    
    % compute the NUFFT
    FFT{f} = NUFFT(kxf + 1i*kyf, w(:,f), [0 0], [Y X]);
end

%% reorganize the data
% the goal is to have it in an array of size length(NUFFT) x Nf * Ncoils
% where length(NUFFT) = rpf * Nsamples
d = EchoSignals(:,1:rpf*Nf,:);
d = reshape(d,[rpf*Nsamples,Nf,Ncoils]);

% density compensation
for c = 1:Ncoils  
    d(:,:,c) = d(:,:,c).*sqrt(w);   
end

% data normalization
im = zeros(Y,X,Nf,Ncoils);
for f = 1:Nf
    im(:,:,f,:) = FFT{f}'*d(:,f,:);
end
d = d/max(abs(im(:)));

%% compute and plot the inverse NUFFT of the data
% compute
simple = zeros(Y,X,Nf);
for f = 1:Nf
	simple(:,:,f) = ESP'*(FFT{f}'*d(:,f,:));
end

% choose the frame to plot
plotframe = randi(Nf);

% plot
figure(1)
subplot(1,2,1)
imagesc(abs(simple(:,:,plotframe)))
title('inverse NUFFT')
axis square

% initialize the plot for the solution of Chambolle-Pock
subplot(1,2,2)
img = imagesc(abs(simple(:,:,plotframe)));
title('Chambolle-Pock')
axis square
ttl = sgtitle(sprintf('frame %d of %d\niteration %d of %d',plotframe,Nf,0,iterations));
drawnow

%% the Chambolle-Pock algorithm
% initialization
y      = { zeros(size(d)); zeros(Y,X,Nf) }; % the variable in the co-domain of K
x_new  = { simple; simple };                % the main variable in the domain of K
u      = x_new;                             % the auxiliary variable in the domain of K
Ku     = { zeros(size(d)); zeros(Y,X,Nf) }; % K u(i)
Kadjy  = { zeros(Y,X,Nf);  zeros(Y,X,Nf) }; % K* y(i+1)
argf   = { zeros(size(d)); zeros(Y,X,Nf) }; % argument of prox_{sigma f*}
argg   = { zeros(Y,X,Nf);  zeros(Y,X,Nf) }; % argument of prox_{tau g}

% soft thresholding
soft   = @(x, t) sign(x).*max(abs(x) - t, 0);

% evaluate objective?
objval = true;
if objval
    objective   = Inf(iterations,1);
    consistency = Inf(iterations,1);
    lowrank     = Inf(iterations,1);
    sparse      = Inf(iterations,1);
    
    figure(2)
    subplot(3,2,1)
    cons = semilogy(consistency);
    title('consistency')

    subplot(3,2,3)
    lowr = semilogy(lowrank);
    title('low-rank')

    subplot(3,2,5)
    spar = semilogy(sparse);
    title('sparse')

    subplot(3,2,[2 4 6])
    obje = semilogy(objective);
    title('objective = consistency + low-rank + sparse')
end

% iterations
for i = 1:iterations
   
    fprintf('Iteration %d of %d\n',i,iterations)

    % keep the solution from the previous iteration
    x_old = x_new;
    
    % precompute the argument of prox_{sigma f*}
    % argf = y(i) + sigma K u(i)
    for f = 1:Nf
        Ku{1}(:,f,:) = FFT{f}*(ESP*(u{1}(:,:,f) + u{2}(:,:,f)));
    end
    Ku{2}(:,:,1:end-1) = diff(u{2},1,3);
    Ku{2}(:,:,end) = zeros(Y,X);
    argf{1} = y{1} + sigma*Ku{1};
    argf{2} = y{2} + sigma*Ku{2};
    
    % apply prox_{sigma f*}
    % y(i+1) = prox_{sigma f*}( argf )
    y{1} = argf{1} - sigma*(argf{1} + d)/(1 + sigma);
    y{2} = argf{2} - sigma*soft(argf{2}/sigma, lambdaS/sigma);
    
    % precompute the argument of prox_{tau g}
    % argg = x(i) - tau K* y(i+1)
    Kadjy{1} = zeros(Y,X,Nf);
    for f = 1:Nf
        Kadjy{1}(:,:,f) = ESP'*(FFT{f}'*y{1}(:,f,:));
    end
    Kadjy{2} = Kadjy{1};
    Kadjy{2}(:,:,1) = Kadjy{2}(:,:,1) - y{2}(:,:,1);
    Kadjy{2}(:,:,2:end-1) = Kadjy{2}(:,:,2:end-1) - diff(y{2}(:,:,1:end-1),1,3);
    Kadjy{2}(:,:,end) = Kadjy{2}(:,:,end) + y{2}(:,:,end-1);
    argg{1} = x_old{1} - tau*Kadjy{1};
    argg{2} = x_old{2} - tau*Kadjy{2};
    
    % apply prox_{tau g}
    % x(i+1) = prox_{tau g}( argg )
    [ U, S, V ] = svd(reshape(argg{1},[X*Y Nf]),'econ');
    x_new{1} = reshape(U*diag(soft(diag(S),S(1,1)*tau*lambdaL))*V',[Y X Nf]);
    x_new{2} = argg{2};
    
    % update the auxiliary variable
    % u(i+1) = x(n+1) + theta ( x(n+1) - x(n) )
    u{1} = x_new{1} + theta*(x_new{1} - x_old{1});
    u{2} = x_new{2} + theta*(x_new{2} - x_old{2});
    
    % evaluate the objective function
    if objval
        Kx = zeros(size(d));
        for f = 1:Nf
        	Kx(:,f,:) = FFT{f}*(ESP*(x_new{1}(:,:,f) + x_new{2}(:,:,f)));
        end
        consistency(i) = 0.5*norm(d(:)-Kx(:))^2;
        ssigma = svd(reshape(x_new{1},[X*Y Nf]),'econ');
        lowrank(i) = lambdaL * ssigma(1) * norm(ssigma,1);
        
        Kx = zeros(Y,X,Nf);
        Kx(:,:,1:end-1) = diff(x_new{2},1,3);
        Kx(:,:,end) = -x_new{2}(:,:,end);
        
        sparse(i)  = lambdaS * norm(Kx(:),1);
        objective(i) = consistency(i) + lowrank(i) + sparse(i);
    end
    
    % build the solution
    solution = abs(x_new{1} + x_new{2});
    
    % plot the solution
    img.CData = solution(:,:,plotframe);
    ttl.String = sprintf('frame %d of %d\niteration %d of %d',plotframe,Nf,i,iterations);
    drawnow
    
    % plot the objective
    if objval
        cons.YData = consistency;
        lowr.YData = lowrank;
        spar.YData = sparse;
        obje.YData = objective;
        drawnow
    end
end

%% save the data
figure

% prepare the time axis
TR = 0.015; % suppose TR = 0.015 s
timeperframe = rpf * TR;
timeaxis = cropsamples * TR + (0 : Nf - 1)*timeperframe;

% save the data
[~,~] = mkdir(recofold);
save([recofold, 'real_', datestr(clock,30)],...
    'iterations', 'TR', 'rpf', 'Nf', 'cropsamples', 'timeaxis', 'solution', 'simple')

% reconstructed frame
s = subplot(2,1,1);
imagesc(abs(solution(:,:,plotframe)))
hold on
colorbar
axis square
title({sprintf('reconstructed frame %d of %d',plotframe,Nf),'right-click to show a perfusion curve, close the figure to end'})

% perfusion curve
while 1
    
    try
        [x, y] = ginput(1);
        x = round(x(1));
        y = round(y(1));
        x = max(1,min(x,X));
        y = max(1,min(y,Y));
    catch
        fprintf('Mischief managed!\n')
        break
    end
    imagesc(s,abs(solution(:,:,plotframe)))
    scatter(s,x,y,'r')
    
    subplot(2,1,2)
    yyaxis left
    plot(timeaxis,squeeze(abs(solution(y,x,:))))
    yyaxis right
    plot(timeaxis,squeeze(abs(simple(y,x,:))))
    xlabel('time / s')
    title(sprintf('perfusion curve at [%d,%d]',x,y))
    legend('L+S','inverse NUFFT')
    ax = gca;
    ax.XGrid = 'on';
    ax.XMinorGrid = 'on';
    xlim([min(timeaxis), max(timeaxis)])
    
end