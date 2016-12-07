close all;
clear all;

file = 'D:\Dokumente\grabbed_data0\scale4\mnist_0_scale04_0550.aedat';
% Load file
[allAddr,allTs]=loadaerdat(file);
% Convert to coordinates, time and event type
[x_coord, y_coord, allTsnew, on_off] = dvsAER2coordinates(allTs, allAddr);

parameters;

% Iterate over events
timewindow_us = 20000;
outputU = zeros(128);
outputV = zeros(128);
outputFilters = zeros(128,128,length(filters));
timeUpdate = uint32(zeros(128));

for i=1:length(x_coord)
    % Current Event
    currX = x_coord(i);
    currY = y_coord(i);
    currT = allTsnew(i);
    currOF = on_off(i);
    
    
    % Find all events in the past
    idx = allTsnew < currT & allTsnew >= currT-timewindow_us;
    windowX = x_coord(idx);
    windowY = y_coord(idx);
    windowT = allTsnew(idx);
    windowOF = on_off(idx);
    
    % For every event in timeframe, convolute
    filterRes(1:length(filters)) = 0;
    for j = 1:length(windowX)
        % Convert time to time in filter space
        tRel = round(abs(windowT(j)-currT)/timewindow_us*(length(times)-1))+1;
        % Center on current event
        xFilter = currX - windowX(j) +  halfGarborRange+1;
        yFilter = currY - windowY(j) +  halfGarborRange+1;
        if(xFilter < 1 ||xFilter > 2*halfGarborRange+1 || yFilter < 1 || yFilter > 2*halfGarborRange+1)
            continue;
        end
        % Convert on/off to -1 and 1
        % ON: 0, OFF: 1
        if(windowOF(j) == 0)
            signal = 1;
        else
            signal = -1;
        end
        % Apply all spartil/temporal filters
        for k=1:length(filters)
            filterRes(k) = filterRes(k) + filters(k).combined(yFilter,xFilter,tRel)*signal;
        end
    end
    
    % Convert filter answer to uv-vector representing the flow direction
    uv = [0 0];
    for k=1:length(filters)
        a = filters(k).G.angle;
        uv = uv + filterRes(k)*[cos(a) -sin(a)];
    end
    % Delete old output
    outputU(timeUpdate < currT-timewindow_us) = 0;
    outputV(timeUpdate < currT-timewindow_us) = 0;
    outputFilters(repmat(timeUpdate < currT-timewindow_us,1,1,length(filters))) = 0;
    
    outputU(currY,currX) = uv(1);
    outputV(currY,currX) = uv(2);
    outputFilters(currY,currX,:) = filterRes;
    timeUpdate(currY,currX) = currT;
    
    % Display every nth frame
    if(mod(i,100) == 0)
        figure(1);
        imagesc(outputU.^2 + outputV.^2);
        figure(2);
        imagesc(outputU);
        figure(3);
        imagesc(outputV);
        figure(4);
        [X,Y] = meshgrid(1:128,1:128);
        quiver(X(:),Y(:),outputU(:),outputV(:),3);
        set(gca,'Ydir','reverse');
        title('Optic flow');
        drawnow;
    end
end