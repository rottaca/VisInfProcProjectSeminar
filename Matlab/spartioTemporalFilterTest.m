close all;
clear all;

%general
times = linspace(0,1,100)';
garborSize = 31;
garborRes = 0.5;

% Parameters for Garbor Filters
thetaGarbor = 25;
% Frequenzy for which the filter is sensitive
f0 = 0.08;
% Prefeered orientation
angle = 45/360*2*pi();
% Compute fx and fy
fx0 = cos(angle)*f0;
fy0 = sin(angle)*f0;

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
[Godd, Geven, X, Y] = constructGarborFilter(thetaGarbor, fx0, fy0, garborSize,garborRes);
tempoBi = constructTemporalBiFilter(thetaBi1,uBi1,s1,thetaBi2,uBi2,s2,times);
tempoMono = constructTemporalMonoFilter(thetaMono,uMono,times);
combined = combineFilters(Geven,Godd,tempoBi,tempoMono);

figure;
surf(X,Y,Godd);
rotate3d on;
title('G_{odd}');
figure;
surf(X,Y,Geven);
rotate3d on;
title('G_{even}');
figure;
hold on;
plot(times,tempoBi);
plot(times,tempoMono);
legend('Biphasic','monophasic');
title('Temporal filters');


figure;
colormap jet;
s = slice(combined/max(combined(:)),1:2:garborSize/garborRes,1:2:garborSize/garborRes,1:15:100);
for n=1:length(s)
    set(s(n), 'EdgeColor', 'none');
    set(s(n),'alphadata',(abs(get(s(n),'cdata'))>0.1)*0.6,'facealpha','flat')
end
title('Sliced spartial-temporal filter, regions with to little responce removed');
xlabel('x');
ylabel('y');
zlabel('t');
axis equal;