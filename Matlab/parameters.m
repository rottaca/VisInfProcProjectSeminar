% Compute speed sensitivity
P1 = [63 69];
P2 = [59 63];
d = norm(P1-P2);
dt = 0.18;
speed_per_sec = d/dt;
filterSpeed = @(dx,dt,windowEnd, windowTime) dx/dt*windowEnd/windowTime;
% Choosen parameters
% Timewindow: 0.135 s
% f0: 0.2
% uBi1 = 0.23
% temporal filter range: 0 - 0.7 s
% Selective speed: 25 px/s
% Max Energy 0.45

% Timewindow: 0.18 s
% f0: 0.15
% uBi1 = 0.23
% temporal filter range: 0 - 0.7 s
% Selective speed: 25 px/s
% Max Energy: 1.2 


%general
times = linspace(0,0.7,100)';
halfGarborRange = 12;

% Parameters for Garbor Filters
sigmaGabor = 25;
% Frequency for which the filter is sensitive
% Higher frequencies -> lower speed selectivity but higher precision
f0 = 0.2;
% Prefeered orientations
angles = 45.0;
% Timewindow
timewindow_us = 84000;

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
    f0, angles,times, halfGarborRange, s1,s2,...
    uBi1, uBi2, sigmaBi1, sigmaBi2,uMono, sigmaMono);
