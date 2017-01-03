close all;
clear all;
parameters;

figure;
hold on;
plot(times,tBi);
plot(times,tMo);
legend('Biphasic','monophasic');
title('Temporal filters');

[X,Y,Z] = meshgrid(-halfGarborRange:halfGarborRange,-halfGarborRange:halfGarborRange,times);

for f = filters
    figure;
    colormap jet;
    % Combine the 4 spatial temporal filters
    %combined = f.combined.EvenBi + f.combined.OddMono;
    % Rightwards
    %combined = f.combined.OddMono - f.combined.EvenBi;
    %combined = f.combined.OddBi + f.combined.EvenMono;
    % Leftwards
    combined = f.combined.OddBi - f.combined.EvenMono;
    %combined = f.combined.OddMono + f.combined.EvenBi;
    
    s = slice(X,Y,Z,combined/max(combined(:)),-halfGarborRange:halfGarborRange,-halfGarborRange:halfGarborRange,times(1:2:end));
    for n=1:length(s)
        set(s(n), 'EdgeColor', 'none');
        set(s(n),'alphadata',(abs(get(s(n),'cdata'))>0.05)*0.6,'facealpha','flat')
    end
    title(['Spartial-temporal filter, angle=' num2str(f.G.angle*360/(2*pi()))]);
    xlabel('x');
    ylabel('y');
    zlabel('t');
    %axis equal;
    axis([-halfGarborRange halfGarborRange -halfGarborRange halfGarborRange 0 times(end)]);
    rotate3d on;
    
    figure;
    surf(X(:,:,1),Y(:,:,1),f.G.Go);
    axis tight;
    rotate3d on;
    title(['G_{odd}, angle=' num2str(f.G.angle*360/(2*pi()))]);
    figure;
    surf(X(:,:,1),Y(:,:,1),f.G.Ge);
    axis tight;
    rotate3d on;
    title(['G_{even}, angle=' num2str(f.G.angle*360/(2*pi()))]);
    
end