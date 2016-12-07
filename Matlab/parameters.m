
%general
times = linspace(0,1,100)';
halfGarborRange = 15;

% Parameters for Garbor Filters
thetaGarbor = 25;
% Frequenzy for which the filter is sensitive
f0 = 0.08;
% Prefeered orientations
angles = 0:45:360;

% Parameters for bi-phasic temporal filter
s1 = 1/2.;
s2 = 3/4.;
uBi1 = 0.2;
uBi2 = 2*uBi1;
thetaBi1 = uBi1/3.;
thetaBi2 = 3/2.*thetaBi1;

% Parameters for mono-phasic temporal filter
uMono = 0.28;%1/5.*(1 + uBi1*sqrt(36 + 10*log(s1/s2)));
thetaMono = uMono/3.;

% Compute individual filters 
[filters, tBi, tMo] = constructSpartialTemporalFilterbank(thetaGarbor, ...
    f0, angles,times, halfGarborRange, s1,s2,...
    uBi1, uBi2, thetaBi1, thetaBi2,uMono, thetaMono);
