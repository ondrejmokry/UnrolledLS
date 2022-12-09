function [h,dh_dx]=Kety_with_derivatives(x, t)
% h- impulse residue function of model Kety with derivatives
% x=abs(x);
Ts=t(2)-t(1);
Ktrans=x(1);
ve=x(2);

h=Ktrans*exp(-(Ktrans/ve)*t);
% t_pul=t(end)/2;
% h(t>t_pul)=0;
h=h*Ts;
% NaNy=sum(isnan(h))
% Infy=sum(isinf(h))
% x
% dervatives of aaTH
if nargout>1
    dh_dKtrans =exp(-Ktrans/ve*t).*(1-Ktrans/ve*t);
    dh_dve =Ktrans^2/ve^2*t.*exp(-Ktrans/ve*t);
    dh_dx=[dh_dKtrans dh_dve];
%     dh_dx(floor(length(t)/2):end)=0;
    dh_dx=dh_dx*Ts;
end
end