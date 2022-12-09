%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                         DCATH_v5_complete.m
%                       Hana Valkova, ID 125086
%                                c2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Funkce pro model TRF - model AATH na zaklade matematickeho popisu modelu
% DCATH - dodany vedoucim prace
% [t TRF]=DCATH_v5_complete(p,t)            
%______________________________________________________________________
% t   - casova osa
% p   - vektor parametru
% TRF - výsledná køivka TRF
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [TRF]=DCATH_v5_complete(p,t,Ts)

F=p(1);
E=p(2);
ve=p(3);
Tc=p(4);
sigma=Ts/2;
Kep=E*F/ve;

N=normcdf(0,Tc,sigma);
erf2=@(x)erf3(x,N);

Hv=1-(erf2((t-Tc)/(sqrt(2)*sigma))+erf2(Tc/(sqrt(2)*sigma)));

Hp=E*exp(1/2*Kep^2*sigma^2+Kep*(Tc-t)).*(erf2((t-Tc)/(sqrt(2)*sigma)-Kep*sigma/sqrt(2))+erf2(Tc/(sigma*sqrt(2))+Kep*sigma/sqrt(2)));
Hp(isnan(Hp))=0;
h=Hp+Hv;
h=h*F;
TRF=h;
end

function h=erf3(x,N)
y=sqrt(2)*x;
h=(normcdf(y)-0.5)/(1-N);
end