close all;
clear all;

file = 'D:\Dokumente\grabbed_data0\scale4\mnist_0_scale04_0550.aedat';
% Load file
[allAddr,allTs]=loadaerdat(file);
% Convert to coordinates, time and event type
[x_coord, y_coord, allTsnew, on_off] = dvsAER2coordinates(allTs, allAddr);
disp(['Event timeframe: ' num2str(double(allTsnew(end))/1000000) ' sek']);

timewindow_us = 30000;
viewdelta = 100;
time = allTsnew(1);
while(time < allTsnew(end))
    idx = allTsnew <= time & allTsnew > time-timewindow_us;
    x = x_coord(idx);
    y = y_coord(idx);
    t = allTsnew(idx);
    of = on_off(idx);
    
    img = zeros(128);
    for j=1:length(x)
        img(y(j),x(j)) = img(y(j),x(j))+(of(j)*2-1);
    end
    
    figure(1);
    imagesc(img,[-1,1]);
    figure(2);
    scatter3(x,y,t-time,10,of);
    axis tight;
    xlim([1 128]);
    ylim([1 128]);
    zlim([-timewindow_us 0 ]);
    drawnow;
    time = time + viewdelta
end