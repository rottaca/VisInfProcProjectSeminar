close all;
clear all;
parameters;
Buffer1 = init3DBuffer(128,128,100);
Buffer2 = init3DBuffer(128,128,100);

% Rightwards
%combined1 = filters(1).combined.OddMono - filters(1).combined.EvenBi;
%combined2 = filters(1).combined.OddBi + filters(1).combined.EvenMono;
% Leftwards
combined1 = filters(1).combined.OddBi - filters(1).combined.EvenMono;
combined2 = filters(1).combined.OddMono + filters(1).combined.EvenBi;

% Buffer = convolute3D(Buffer,filter,10,10); 
% Buffer.W = mod(Buffer.W,size(Buffer.buff,3))+1;
% [Buffer, res] = read3DConvolutionRes(Buffer);
% Buffer = convolute3D(Buffer,filter,15,15);  
% Buffer.W = mod(Buffer.W,size(Buffer.buff,3))+1;
% [Buffer, res] = read3DConvolutionRes(Buffer);
% imagesc(res);
% 
% figure;
% imagesc(squeeze(Buffer.buff(:,10,:)));

file = 'D:\Dokumente\grabbed_data0\scale4\mnist_0_scale04_0550.aedat';
% Load file
[allAddr,allTs]=loadaerdat(file);
% Convert to coordinates, time and event type
[x_coord, y_coord, allTsnew, on_off] = dvsAER2coordinates(allTs, allAddr);

% Iterate over events
timewindow_us = 30000;

currTime = allTsnew(1);
timeRes = timewindow_us/length(times);
for i=1:length(x_coord)
    % Current Event
    currX = x_coord(i);
    currY = y_coord(i);
    currT = allTsnew(i);
    currOF = on_off(i);
    
    % ON: 0, OFF: 1
    signal = 0;
    if(currOF == 0)
        signal = 1;
    else
        continue;
    end
        
    deltaT = currT - currTime;
    timeSlotsToSkip = floor(double(deltaT)/timeRes);
    for j = 1:timeSlotsToSkip
        Buffer1.W = mod(Buffer1.W,size(Buffer1.buff,3))+1;
        [Buffer1, ~] = read3DConvolutionRes(Buffer1);
        Buffer2.W = mod(Buffer2.W,size(Buffer2.buff,3))+1;
        [Buffer2, ~] = read3DConvolutionRes(Buffer2);
    end
    currTime = currT;
    
    Buffer1 = convolute3D(Buffer1,combined1,currX,currY); 
    [Buffer1, res1] = read3DConvolutionRes(Buffer1);
    Buffer2 = convolute3D(Buffer2,combined2,currX,currY); 
    [Buffer2, res2] = read3DConvolutionRes(Buffer2);
        
    % Display every nth frame
    %if mod(i,10) == 0
        figure(1);
        energy = res1.^2 + res2.^2;
        imagesc(energy);
        max(energy(:))
        colormap jet;
        hold on;
        disp(['events visible: ' num2str(sum(allTsnew <= currT & allTsnew >= currT-timewindow_us & 1:length(x_coord) <= i))]);
        xSlot = x_coord(allTsnew <= currT & allTsnew >= currT-timewindow_us & 1:length(x_coord) <= i);
        ySlot = y_coord(allTsnew <= currT & allTsnew >= currT-timewindow_us & 1:length(x_coord) <= i);
        scatter(xSlot,ySlot,'filled','g');
        axis([1 128 1 128]);
        set(gca,'Ydir','reverse');
        figure(2);
        imagesc(squeeze(Buffer1.buff(64,:,[Buffer1.R:size(Buffer1.buff,3) 1:Buffer1.R-1])));
        drawnow;
   % end
    %break;
    
end