%close all;
clear all;
parameters;

figure(1);
cla;
hold on;
plot(times,tBi);
plot(times,tMo);
legend('Biphasic','monophasic');
title('Temporal filters');

[X,Y,Z] = meshgrid(-halfGarborRange:resolutionSpatial:halfGarborRange,-halfGarborRange:resolutionSpatial:halfGarborRange,times);

figureIdx = 2;
for f = filters
    %figure;
    %colormap jet;
    % Combine the 4 spatial temporal filters
    %combined = f.combined.EvenBi + f.combined.OddMono;
    % Rightwards
    %combined = f.combined.OddMono - f.combined.EvenBi;
    %combined = f.combined.OddBi + f.combined.EvenMono;
    % Leftwards
    %combined = f.combined.OddBi - f.combined.EvenMono;
    combined = f.combined.OddMono + f.combined.EvenBi;
    
%     s = slice(X,Y,Z,combined/max(combined(:)),-halfGarborRange:resolutionSpatial:halfGarborRange,-halfGarborRange:resolutionSpatial:halfGarborRange,times(1:length(times)/50:end));
%     for n=1:length(s)
%         set(s(n), 'EdgeColor', 'none');
%         set(s(n),'alphadata',(abs(get(s(n),'cdata'))>0.05)*0.6,'facealpha','flat')
%     end
%     title(['Spartial-temporal filter, angle=' num2str(f.G.angle*360/(2*pi()))]);
%     xlabel('x');
%     ylabel('y');
%     zlabel('t');
%     %axis equal;
%     axis([-halfGarborRange halfGarborRange -halfGarborRange halfGarborRange 0 times(end)]);
%     rotate3d on;
    
%     figure;
%     surf(X(:,:,1),Y(:,:,1),f.G.Go);
%     axis tight;
%     rotate3d on;
%     title(['G_{odd}, angle=' num2str(f.G.angle*360/(2*pi()))]);
%     figure;
%     surf(X(:,:,1),Y(:,:,1),f.G.Ge);
%     axis tight;
%     rotate3d on;
%     title(['G_{even}, angle=' num2str(f.G.angle*360/(2*pi()))]);
        sliceXT = squeeze(combined(halfGarborRange,:,:));
        fft_sliceXT = fftshift(abs(fft2(sliceXT)));
        [fxMax, ftMax] = find(max(fft_sliceXT(:)) == fft_sliceXT, 1);
        
        sliceYT = squeeze(combined(:,halfGarborRange,:));
        fft_sliceYT = fftshift(abs(fft2(sliceYT)));
        [fyMax, ftMax2] = find(max(fft_sliceYT(:)) == fft_sliceYT, 1);
        if(fft_sliceYT(fyMax,ftMax2) > fft_sliceXT(fxMax,ftMax)) 
            ftMax = ftMax2;
        end
        
        offset = floor(size(fft_sliceXT)/2)+1;
        % Shift coordinates, so that origin is at image center
        fxMax = fxMax - offset(1);
        fyMax = fyMax - offset(1);
        ftMax = ftMax - offset(2);
        
        % Covert to frequency
        fxMax = fxMax/size(sliceXT,1);
        fyMax = fyMax/size(sliceXT,1);
        ftMax = ftMax/size(sliceXT,2);
        % Compute speed in x and y direction
        sX = -ftMax/(fxMax^2+fyMax^2)*fxMax;
        sY = -ftMax/(fxMax^2+fyMax^2)*fyMax;
        % Compute |S|
        s = sqrt(sX^2+sY^2);
        
        figure(figureIdx),figureIdx= figureIdx+1; imagesc(sliceXT)
        figure(figureIdx),figureIdx= figureIdx+1; imagesc(fft_sliceXT)
        figure(figureIdx),figureIdx= figureIdx+1; imagesc(sliceYT)
        figure(figureIdx),figureIdx= figureIdx+1; imagesc(fft_sliceYT)
        
        speed = s/(timewindow_us/1000000/length(times));
        fprintf('Speed in pixel/s: %d \n',speed);

end