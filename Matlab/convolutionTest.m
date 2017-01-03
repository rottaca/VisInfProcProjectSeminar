close all;
clear all;
parameters;
Buffer1 = init3DBuffer(128,128,length(times));
Buffer2 = init3DBuffer(128,128,length(times));

% Rightwards
combined2 = filters(1).combined.OddMono - filters(1).combined.EvenBi;
combined1 = filters(1).combined.OddBi + filters(1).combined.EvenMono;
% Leftwards
%combined1 = filters(1).combined.OddBi - filters(1).combined.EvenMono;
%combined2 = filters(1).combined.OddMono + filters(1).combined.EvenBi;

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

slotTime = allTsnew(1);
timeRes = timewindow_us/length(times);
maxEnergy = [];%zeros(1,length(x_coord));
for i=1:length(x_coord)
    % Current Event
    currX = 129-x_coord(i);
    currY = 129-y_coord(i);
    currT = allTsnew(i);
    currOF = on_off(i);
    %disp(['Event ' num2str(i) ' time: ' num2str(currT)]);
    
    % ON: 0, OFF: 1
    signal = 0;
    if(currOF == 0)
        signal = 1;
%         disp('Taken');
    else
%         disp('Discharged');
        continue;
    end
    
    
    %disp(['Done: ' num2str(100*double(i)/length(x_coord)) '%']);
        
    deltaT = currT - slotTime;
    timeSlotsToSkip = floor(double(deltaT)/timeRes);
    for j = 1:timeSlotsToSkip
%         disp(['Slots to skip: ' num2str(timeSlotsToSkip) ' startTime: ' num2str(slotTime) ' delta: ' num2str(deltaT)]);
        [Buffer1, res1] = read3DConvolutionRes(Buffer1);
        [Buffer2, res2] = read3DConvolutionRes(Buffer2);
    % Display every nth frame
    %if mod(i,10) == 0
        figure(1);
         energy = res1.^2 + res2.^2;
        imagesc(energy);
        caxis([0 2]);
        maxEnergy(end+1) = max(energy(:));
        max(energy(:));
        colormap jet;
        hold on;
        windowIdices = allTsnew <= currT & allTsnew >= currT-timewindow_us & 1:length(x_coord) <= i & on_off == currOF;
        %disp(['events visible: ' num2str(sum(windowIdices))]);
        xSlot = x_coord(windowIdices);
        ySlot = y_coord(windowIdices);
        c = allTsnew(windowIdices);
        c = c - min(c(:));
        c = round(double(c)/max(c(:))*255.0);
        c = squeeze(ind2rgb(uint8(c),gray(256)));
        scatter(xSlot,ySlot,10,c,'filled');
        axis([1 128 1 128]);
        set(gca,'Ydir','reverse');
        figure(2);
        imagesc(squeeze(Buffer1.buff(64,:,[Buffer1.W+1:size(Buffer1.buff,3) 1:Buffer1.W])),[-0.02 0.02]);
        title('Buffer 1');
        axis equal;
        axis tight;
        %caxis([-1 1]);
        figure(3);
        imagesc(squeeze(Buffer2.buff(64,:,[Buffer2.W+1:size(Buffer2.buff,3) 1:Buffer2.W])),[-0.02 0.02]);
        title('Buffer 2');
        axis equal;
        axis tight;
        %caxis([-1 1]);
        drawnow;
        slotTime = slotTime + timeRes;
    end
    
    Buffer1 = convolute3D(Buffer1,combined1,currX,currY); 
    Buffer2 = convolute3D(Buffer2,combined2,currX,currY); 
end
figure;
bar(maxEnergy);