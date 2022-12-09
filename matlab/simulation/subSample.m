function signalOut = subSample(signalIn, subSampFac)
signalIn = signalIn(:)';
origSize = length(signalIn);
newSize = round(origSize/subSampFac);
startPoints = 1:subSampFac:origSize;
signalOut = zeros(1, newSize);
for j = 1:newSize
    if startPoints(j) + subSampFac - 1 <= origSize
        signalOut(j) = mean(signalIn(startPoints(j):startPoints(j)+subSampFac-1));
    else
        signalOut(j) = mean(signalIn(startPoints(j):end));    
    end
end
end

