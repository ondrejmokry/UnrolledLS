% add the package
fileID = fopen('../../packagefold.txt','r');
packagefold = fscanf(fileID,'%s');
fclose(fileID);

addpath(genpath([packagefold, '\ESPIRiT']));

% reshape k and w such that it is a column vector per each frame
if size(k,2) > size(k,1)
    k = transpose(k);
end
if size(w,2) > size(w,1)
    w = transpose(w);
end

% read the number of frames as the smaller dimension of k (or w)
Nf = size(k,2);

% initialize the cell array
FFT = cell(Nf,1);

% construct the operators
for f = 1:Nf
	FFT{f} = NUFFT(k(:,f), w(:,f), shift, imSize);
end