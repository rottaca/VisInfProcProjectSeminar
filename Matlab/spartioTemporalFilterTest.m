close all;
clear all;

%general
times = linspace(0,1,100)';
garborSize = 20;
garborRes = 0.5;

% Parameters for Garbor Filters
thetaGarbor = 25;
% Frequenzy for which the filter is sensitive
f0 = 0.08;
% Prefeered orientations
angles = 0;%0:45:135;

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
    f0, angles,times, garborSize, garborRes, s1,s2,...
    uBi1, uBi2, thetaBi1, thetaBi2,uMono, thetaMono);

figure;
hold on;
plot(times,tBi);
plot(times,tMo);
legend('Biphasic','monophasic');
title('Temporal filters');

for f = filters
    figure;
    colormap jet;
    s = slice(f.combined/max(f.combined(:)),1:2:garborSize/garborRes,1:2:garborSize/garborRes,1:15:100);
    for n=1:length(s)
        set(s(n), 'EdgeColor', 'none');
        set(s(n),'alphadata',(abs(get(s(n),'cdata'))>0.1)*0.6,'facealpha','flat')
    end
    title(['Spartial-temporal filter, angle=' num2str(f.G.angle*360/(2*pi()))]);
    xlabel('x');
    ylabel('y');
    zlabel('t');
    axis equal;
    rotate3d on;
    
    figure;
    surf(f.G.Go);
    axis tight;
    rotate3d on;
    title(['G_{odd}, angle=' num2str(f.G.angle*360/(2*pi()))]);
    figure;
    surf(f.G.Ge);
    axis tight;
    rotate3d on;
    title(['G_{even}, angle=' num2str(f.G.angle*360/(2*pi()))]);
    
end