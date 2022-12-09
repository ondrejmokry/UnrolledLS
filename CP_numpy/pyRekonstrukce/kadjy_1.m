Kadjy2(:,:,1) = Kadjy2(:,:,1) - y2(:,:,1);
% Kadjy2(:,:,2:end-1) = Kadjy2(:,:,2:end-1) - diff(y2(:,:,1:end-1),1,3);
% Kadjy2(:,:,end) = Kadjy2(:,:,end) + y2(:,:,end-1);