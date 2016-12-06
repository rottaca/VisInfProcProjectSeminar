function [x_coord, y_coord, allTsnew, on_off] = dvsAER2coordinates(allTs, allAddr)
% convert allTs and allAddr as read by loadaerdat to 
% [x_coord, y_coord, allTsnew, on_off]
% (C)ST,TB 2014 Ulm University

numEvents=numel(allTs);

x_coord = zeros(1, numEvents);
y_coord = zeros(1, numEvents);
allTsnew = zeros(1, numEvents);
on_off = zeros(1, numEvents);

fprintf('dvsAER2coordinates for %d events      ', size(allAddr,1) );

showpercent = 10; 

count=0;
tic
for currentEvent=1:numEvents(1)
    curUint = allAddr(currentEvent);
    
%     if (currentEvent/numEvents*100>showpercent),
%         fprintf('\b\b\b\b\b %03.0d ',showpercent); 
%         showpercent = showpercent + 20; 
%     end
    
    %%
    
    onOffBit = mod(curUint, 2)   ; % last byte
    curAddr = double(bitshift(curUint, -1, 'uint32')); % first bytes except last
    
    column = mod(curAddr, 128)   ;
    row = (curAddr - column) / 128   ;
    column = 128-column  ; 
    row = 128-row   ;
    % Test
%     column = 1+column  ; 
%     row = 1+row   ;
    
    if (column>0) && (column<129) && (row>0) && (row<129),
        count = count + 1;
        x_coord(count) = column;
        y_coord(count) = row;
        allTsnew(count) = allTs(currentEvent);        
        on_off(count) = onOffBit;
    else
        fprintf('row=%d, column=%d\n', row, column);
    end
      
end
fprintf('%d/%d events sorted out\n', numEvents-count, numEvents);
toc

% strip to valid length count
x_coord = x_coord(1:count);
y_coord = y_coord(1:count);
allTsnew = allTsnew(1:count) - allTsnew(1);
on_off = on_off(1:count);

fprintf('\n'); 