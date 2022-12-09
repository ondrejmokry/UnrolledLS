function [aif_triexponencial]=AIF_triexpG(A, B, C, tau1, tau2, tau3, beta, Ts, N)
%% Sum of 2 decreasing exponentials and one gamma-variate function

% Ts = 1;
% N = 600;
% 
% T = 0.17046;
% T2 = 0.365;
% 
% C= 0.15;
% D= 5.7;
% 
% A = 1;
% B = 0.69;

t=(0:(N-1))*Ts;

s = zeros(2,int32(N));

s(1,:) = t.^beta .* A.*(exp(-tau1*t));
s(2,:) = t.^beta .* B.*(exp(-tau2*t));
s(3,:) = t.^beta .* C.*(exp(-tau3*t));

% s(1,:) = A.*(tau1.^(-t));
% s(2,:) = B.*(tau2.^(-t));
% s(3,:) = C.*(tau3.^(-t));

aif_triexponencial = sum(s);
end