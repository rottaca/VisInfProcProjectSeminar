
%general
times = linspace(0,1,100)';
halfGarborRange = 15;

% Parameters for Garbor Filters
sigmaGabor = 25;
% Frequenzy for which the filter is sensitive
f0 = 0.08;
% Prefeered orientations
angles = 0;

% Parameters for bi-phasic temporal filter
s1 = 1/2.;
s2 = 3/4.;
uBi1 = 0.2;
uBi2 = 2*uBi1;
sigmaBi1 = uBi1/3.;
sigmaBi2 = 3/2.*sigmaBi1;

% Parameters for mono-phasic temporal filter
uMono = 0.28;%1/5.*(1 + uBi1*sqrt(36 + 10*log(s1/s2)));
sigmaMono = uMono/3.;

% Compute individual filters 
[filters, tBi, tMo] = constructSpatialTemporalFilterbank(sigmaGabor, ...
    f0, angles,times, halfGarborRange, s1,s2,...
    uBi1, uBi2, sigmaBi1, sigmaBi2,uMono, sigmaMono);
