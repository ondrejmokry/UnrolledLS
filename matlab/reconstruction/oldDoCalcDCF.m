% Numerical MRI Simulation Package
% Version 1.2  - https://sourceforge.net/projects/mrilab/
%
% The MRiLab is a numerical MRI simulation package. It has been developed to 
% simulate factors that affect MR signal formation, acquisition and image 
% reconstruction. The simulation package features highly interactive graphic 
% user interface for various simulation purposes. MRiLab provides several 
% toolboxes for MR researchers to analyze RF pulse, design MR sequence, 
% configure multiple transmitting and receiving coils, investigate B0 
% in-homogeneity and object motion sensitivity et.al. The main simulation 
% platform combined with these toolboxes can be applied for customizing 
% various virtual MR experiments which can serve as a prior stage for 
% prototyping and testing new MR technique and application.
%
% Author:
%   Fang Liu <leoliuf@gmail.com>
%   University of Wisconsin-Madison
%   April-6-2014
%
% Edited by:
%   Ondrej Mokry <ondrej.mokry@mensa.cz>
%   Brno University of Technology
%   February-17-2021
% _________________________________________________________________________
% Copyright (c) 2011-2014, Fang Liu <leoliuf@gmail.com>
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are 
% met:
% 
%     * Redistributions of source code must retain the above copyright 
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright 
%       notice, this list of conditions and the following disclaimer in 
%       the documentation and/or other materials provided with the distribution
%       
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
% POSSIBILITY OF SUCH DAMAGE.
% _________________________________________________________________________

function DCF = oldDoCalcDCF(Kx, Ky)
% caluclate density compensation factor using Voronoi diagram

% remove duplicated K space points (likely [0,0]) before Voronoi
K = Kx + 1i*Ky;
[~,m1,n1] = unique(K);
K = K(sort(m1));

% calculate Voronoi diagram
[K2,~,~] = unique(K);
Kx = real(K2);
Ky = imag(K2);
Area = voronoiarea(Kx,Ky);

% use area as density estimate
DCF = Area(n1);

% take equal fractional area for repeated locations (likely [0,0])
n = n1;
while ~isempty(n)
    rep = length(find(n == n(1)));
    if rep > 1
        DCF(n1 == n(1)) = DCF(n1 == n(1))./rep;
    end
    n(n == n(1)) = [];
end

% normalize DCF
% DCF = DCF ./ max(DCF);

function Area = voronoiarea(Kx,Ky)
% caculate area for each K space point as density estimate

Kxy = [Kx,Ky];
% returns vertices and cells of voronoi diagram
[V,C] = voronoin(Kxy);

% compute area of each ploygon
Area = zeros(1,length(Kx));
for j = 1:length(Kx)
    x = V(C{j},1); 
    y = V(C{j},2);
    % remove vertices outside K space limit including infinity vertices from voronoin
    x1 = x;
    y1 = y;
    ind = find((x1.^2 + y1.^2)>0.25);
    x(ind) = []; 
    y(ind) = [];
    % calculate area
    lxy = length(x);
    if lxy > 2
        ind=[2:lxy 1];
        A = abs(sum(0.5*(x(ind)-x(:)).*(y(ind)+y(:))));
    else
        A = 0;
    end
    Area(j) = A;
end