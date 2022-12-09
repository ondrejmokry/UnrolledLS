% the script performs reconstruction from the synthetic DCE-MRI data from
% the simulator
%
% Date: 08/04/2022
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

% save results?
savedata = false;

%% load the data
datafold = [df, '/simulation/'];
datafile = 'SyntheticEchoes_nufft_64';
recofold = [df, '/reconstruction/'];

load([datafold, datafile])
load([datafold, 'sensitivities'])

[Nsamples, ~, Ncoils] = size(EchoSignals);
[Y, X, ~] = size(Sensitivities);
% aif           perfusion curve in N time instants (vector 1 x N)
% Angles        angles under which the data are sampled [rad] (vector 1 x N)
% delay         bolus arrival time [s]
% EchoSignals   the signal (array Nsamples x N x Ncoils)
% FA            flip angle of excitation RF pulses [rad]
% N             number of samples
% Ncoils        number of coil elements
% Nsamples      number of samples of each echo
% PhEncShifts   phase encoding shifts, for radial sampling always zero (vector 1 x N)
% r1            r1 relaxivity of tissue and blood [l/mmol/s]
% Sensitivities sensitivities of the coil elements (array X x Y x Ncoils)
% TR            time interval between two consecutive samples [s]
% X, Y          size of the image

%% crop the beginning
startpoint  = 20; % crop interval [s]
cropsamples = ceil(startpoint/TR);
aif         = aif(cropsamples+1:end);
Angles      = Angles(cropsamples+1:end);
EchoSignals = EchoSignals(:,cropsamples+1:end,:);
N           = N - cropsamples;
PhEncShifts = PhEncShifts(cropsamples+1:end);

%% settings
% radials per frame
rpf = 23;

% number of frames
Nf = 120;
Nf = min(Nf, floor(N/rpf));

% regularizers (parameters of the model)
lambdaS = 1e-7;
lambdaL = 1e-1;

% parameters of the Chambolle-Pock algorithm, part one
theta      = 1;
iterations = 100;

% subsampling factor (for speed-up)
factor = 8;

%% subsample and normalize the sensitivities
% subsample
X = X/factor;
Y = Y/factor;
C = zeros(Y,X,Ncoils);
for c = 1:Ncoils
    C(:,:,c) = imresize(Sensitivities(:,:,c), 1/factor);
end

% normalize
norms = sqrt(sum(conj(C).*C,3));
C = C./norms;

%% precompute the NUFFT operators
w   = zeros(Nsamples*rpf,Nf);
FFT = cell(Nf,1);
for f = 1:Nf
    
    fprintf('Defining the NUFFT for frame %d of %d.\n',f,Nf)
    
    % compute the k-space locations of the samples (taking into account the
    % convention of NUFFT)
    fAngles = Angles((f-1)*rpf+1:f*rpf);
    kx      = ( (-Nsamples/2):(Nsamples/2-1) )'*sin(fAngles);
    ky      = ( (-Nsamples/2):(Nsamples/2-1) )'*cos(fAngles);
    kx      = kx(:)/X;
    ky      = ky(:)/Y;
    
    % compute the density compensation
    % the function DoCalcDCF has been edited such that the areas are not
    % normalized
    w(:,f) = DoCalcDCF(kx, ky)*X*Y;
    
    % eliminate the zeros on the edge of the Voronoi diagram
    for r = 1:rpf
        % the first value per each radial
        w(1+(r-1)*Nsamples,f) = 2*w(2+(r-1)*Nsamples,f) - w(3+(r-1)*Nsamples,f);
        
        % the last value per each radial
        w(r*Nsamples,f) = 2*w(r*Nsamples-1,f) - w(r*Nsamples-2,f);
    end
    
    % compute the NUFFT
    FFT{f} = NUFFT(kx + 1i*ky, w(:,f), [0 0], [Y X]);
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
d = d/(sqrt(X*factor*Y*factor)*factor);

%% compute and plot the inverse NUFFT of the data
% compute
fprintf('Computing the simple solution (inverse NUFFT).\n')
simple = zeros(Y,X,Nf);
for f = 1:Nf
    for c = 1:Ncoils
        simple(:,:,f) = simple(:,:,f) + conj(C(:,:,c)).*(FFT{f}'*d(:,f,c));
    end
end

% trying to get rid of significant imaginary part of the solution
simple = abs(simple);

% choose the frame to plot
plotframe = randi(Nf);

% plot
figure(1)
subplot(2,2,1)
imagesc(abs(simple(:,:,plotframe)))
title('inverse NUFFT')
axis square
colorbar

% initialize the plot for the solution of Chambolle-Pock
subplot(2,2,2)
img = imagesc(abs(simple(:,:,plotframe)));
title('Chambolle-Pock (sum)')
axis square
colorbar

% initialize the plot for the L component
subplot(2,2,3)
img_L = imagesc(abs(simple(:,:,plotframe)));
title('Chambolle-Pock (L)')
axis square
colorbar

% initialize the plot for the L component
subplot(2,2,4)
img_S = imagesc(abs(simple(:,:,plotframe)));
title('Chambolle-Pock (S)')
axis square
colorbar

ttl = sgtitle(sprintf('frame %d of %d\niteration %d of %d',plotframe,Nf,0,iterations));
drawnow

%% estimate the operator norms
% this is done using the (normalized) power method to estimate the largest
% eigenvalue of the operator FFT{f}'*FFT{f}, the square root of which is
% the norm of FFT{f}
fprintf('Computing the operator norms.\n')
normiterations = 100;
normf = NaN(normiterations, Nf);
for f = 1:Nf
    x = simple(:,:,f);
    for i = 1:normiterations
        y = FFT{f}'*(FFT{f}*x);
        normf(i, f) = max(abs(y(:)));
        x = y/normf(i, f);
    end
end

% parameters of the Chambolle-Pock algorithm, part two
sigma = 1/sqrt(4*max(normf(:)) + 4);
tau   = 1/sqrt(4*max(normf(:)) + 4);

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
        for c = 1:Ncoils
            Ku{1}(:,f,c) = FFT{f}*(C(:,:,c).*(u{1}(:,:,f) + u{2}(:,:,f)));
        end
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
        for c = 1:Ncoils
            Kadjy{1}(:,:,f) = Kadjy{1}(:,:,f) + conj(C(:,:,c)).*(FFT{f}'*y{1}(:,f,c));
        end
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
            for c = 1:Ncoils
                Kx(:,f,c) = FFT{f}*(C(:,:,c).*(x_new{1}(:,:,f) + x_new{2}(:,:,f)));
            end
        end
        consistency(i) = 0.5*norm(d(:)-Kx(:))^2;
        ssigma = svd(reshape(x_new{1},[X*Y Nf]),'econ');
        lowrank(i) = lambdaL * ssigma(1) * norm(ssigma,1);
        
        Kx = zeros(Y,X,Nf);
        Kx(:,:,1:end-1) = diff(x_new{2},1,3);
        Kx(:,:,end) = -x_new{2}(:,:,end);
        
        sparse(i) = lambdaS * norm(Kx(:),1);
        objective(i) = consistency(i) + lowrank(i) + sparse(i);
    end
    
    % build the solution
    solution = abs(x_new{1} + x_new{2});
    
    % plot the solution
    img.CData = solution(:,:,plotframe);
    
    % plot the L and S components
    img_L.CData = abs(x_new{1}(:,:,plotframe));
    img_S.CData = abs(x_new{2}(:,:,plotframe));
    
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

%% compute the ground truth
fprintf('Generating the ground truth.\n')
addpath('../simulation')
gt = getImages(startpoint + (rpf*TR) * (0.5 + (0:Nf-1)), 1/factor);

%% compute the version multiplied with the sensitivities
gtc = gt;
for f = 1:Nf
    gtc(:,:,f) = gtc(:,:,f).*norms;
end

%% make a movie
figure('Position',[400 400 1600 400])
t = tiledlayout(1,3);

nexttile
img = imagesc(solution(:,:,1), [min(solution(:)), max(solution(:))]);
title('L + S')
axis square
colorbar

nexttile
img_L = imagesc(abs(x_new{1}(:,:,1)), [min(solution(:)), max(solution(:))]);
title('L')
axis square
colorbar

nexttile
img_S = imagesc(abs(x_new{2}(:,:,1)), [min(solution(:)), max(solution(:))]);
title('S')
axis square
colorbar

ttl = title(t,sprintf('frame 1 of %d',Nf));
    
% F(Nf) = struct('cdata',[],'colormap',[]);
for f = 1:Nf
    
    img.CData = solution(:,:,f);
    img_L.CData = abs(x_new{1}(:,:,f));
    img_S.CData = abs(x_new{2}(:,:,f));
    
    ttl.String = sprintf('frame %d of %d',f,Nf);
    
    % pause(0.1)
    drawnow
    
    % F(f) = getframe(gcf);
end

%% plot the perfusion curves
figure

% prepare the time axis
timeperframe = rpf * TR;
timeaxis = cropsamples * TR + (0 : Nf - 1)*timeperframe;

% save the data
if savedata
    [~,~] = mkdir(recofold);
    save([recofold, 'LS_', datestr(clock,30)],...
        'C', 'cropsamples', 'gt', 'gtc', 'iterations',...
        'lambdaL', 'lambdaS', 'Nf', 'rpf',...
        'sigma', 'simple', 'solution', 'tau', 'timeaxis', 'TR', 'x_new')
end

% reconstructed frame
s = subplot(3,3,1);
imagesc(abs(solution(:,:,plotframe)))
hold on
colorbar
axis square
title({sprintf('reconstructed frame %d of %d',plotframe,Nf),...
    'click to show a perfusion curve, close the figure to end'})

% ground truth frame
subplot(3,3,2);
imagesc(abs(gt(:,:,plotframe)))
hold on
colorbar
axis square
title(sprintf('ground truth frame %d of %d',plotframe,Nf))

% ground truth frame multiplied with the sensitivity norms
subplot(3,3,3);
imagesc(abs(gtc(:,:,plotframe)))
hold on
colorbar
axis square
title({sprintf('ground truth frame %d of %d',plotframe,Nf),...
    'multiplied with the sensitivity norms'})

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
    
    subplot(3,3,4:6)
    yyaxis left
    plot(timeaxis,squeeze(abs(solution(y,x,:))),...
         timeaxis,squeeze(abs(simple(y,x,:))),':',...
         timeaxis,squeeze(abs(gtc(y,x,:))),'--')
    yyaxis right
    plot(timeaxis,squeeze(abs(gt(y,x,:))))
    xlabel('time / s')
    title(sprintf('perfusion curve at [%d,%d]',x,y))
    legend('L+S','inverse NUFFT','scaled ground truth','ground truth')
    ax = gca;
    ax.XGrid = 'on';
    ax.XMinorGrid = 'on';
    xlim([min(timeaxis), max(timeaxis)])
    
    % perfusion curve for the L and S components
    subplot(3,3,7:9)
    plot(timeaxis,squeeze(abs(x_new{1}(y,x,:))))
    hold on
    plot(timeaxis,squeeze(abs(x_new{2}(y,x,:))))
    xlabel('time / s')
    title(sprintf('perfusion curve at [%d,%d]',x,y))
    legend('L','S')
    ax = gca;
    ax.XGrid = 'on';
    ax.XMinorGrid = 'on';
    xlim([min(timeaxis), max(timeaxis)])
    hold off
    
end