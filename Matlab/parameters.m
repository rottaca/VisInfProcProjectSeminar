% Choosen parameters

% Timewindow: 0.30 s
% f0: 0.15
% uBi1 = 0.23
% temporal filter range: 0 - 0.7 s
% Selective speed: 22.244 px/s

% Timewindow: 0.20 s
% f0: 0.15
% uBi1 = 0.23
% temporal filter range: 0 - 0.7 s
% Selective speed: 33.366 px/s

% Timewindow: 0.10 s
% f0: 0.15
% uBi1 = 0.23
% temporal filter range: 0 - 0.7 s
% Selective speed: 66.733 px/s

%general
times = linspace(0,0.7,50)';
halfGarborRange = 500;
resolutionSpatial = 1;

% Parameters for Garbor Filters
sigmaGabor = 25;
% Frequency for which the filter is sensitive
% Higher frequencies -> lower speed selectivity but higher precision
f0 = 0.15;
% Prefeered orientations
angles = 0.0;
% Timewindow
timewindow_us =100000;

% Parameters for bi-phasic temporal filter
s1 = 1/2.;
s2 = 3/4.;
% higher uBi1 -> lower speed selectivity
uBi1 = 0.23;
uBi2 = 2*uBi1;
sigmaBi1 = uBi1/3.;
sigmaBi2 = 3/2.*sigmaBi1;

% Parameters for mono-phasic temporal filter
uMono = 1/5.*uBi1*(1 + sqrt(36 + 10*log(s1/s2)));
sigmaMono = uMono/3.;

% Compute individual filters 
[filters, tBi, tMo] = constructSpatialTemporalFilterbank(sigmaGabor, ...
    f0, angles,times, halfGarborRange,resolutionSpatial, s1,s2,...
    uBi1, uBi2, sigmaBi1, sigmaBi2,uMono, sigmaMono);
