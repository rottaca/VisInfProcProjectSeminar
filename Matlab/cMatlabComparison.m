spartioTemporalFilterTest;

r1 = csvread('G:\filter_right1.txt')';
r2 = csvread('G:\filter_right2.txt')';
l1 = csvread('G:\filter_left1.txt')';
l2 = csvread('G:\filter_left2.txt')';

% Rightwards
l2_ = f.combined.OddMono - f.combined.EvenBi;
l1_ = f.combined.OddBi + f.combined.EvenMono;
% Leftwards
r1_ = f.combined.OddBi - f.combined.EvenMono;
r2_ = f.combined.OddMono + f.combined.EvenBi;

dl1 = abs(squeeze(l1_(13,:,:)) - l1);
dl2 = abs(squeeze(l2_(13,:,:)) - l2);
dr1 = abs(squeeze(r1_(13,:,:)) - r1);
dr2 = abs(squeeze(r2_(13,:,:)) - r2);

mn = min([dl1(:); dl2(:); dr1(:); dr2(:)])
mx = max([dl1(:); dl2(:); dr1(:); dr2(:)])

close all;
figure; imagesc(dl1, [mn mx]);title('l1');
figure; imagesc(dl2, [mn mx]);title('l2');
figure; imagesc(dr1, [mn mx]);title('r1');
figure; imagesc(dr2, [mn mx]);title('r2');

file = 'D:\Dokumente\grabbed_data0\scale4\mnist_0_scale04_0550.aedat';
% Load file
[allAddr,allTs]=loadaerdat(file);
% Convert to coordinates, time and event type
[x_coord, y_coord, allTsnew, on_off] = dvsAER2coordinates(allTs, allAddr);

c = csvread('G:\on_x_y_time.txt');
c_ = [on_off', x_coord', y_coord', allTsnew'];
dc = c_-c;
figure; bar(dc(:,1));title('OnOff');
figure; bar(dc(:,2));title('X');
figure; bar(dc(:,3));title('Y');
figure; bar(dc(:,4));title('Time');

x = csvread('G:\buff_l1_xz_y_64.txt')';
figure;imagesc(x);
x_ = init3DBuffer(128,128,length(times));
x_ = convolute3D(x_,l1_,64,64); 
figure;imagesc(squeeze(x_.buff(64,:,:)));
dx = abs(squeeze(x_.buff(64,:,:))-x);
figure;imagesc(dx);title('Error Ringbuffer');
