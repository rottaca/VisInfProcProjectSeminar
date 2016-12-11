function [ Buffer ] = convolute3D( Buffer, Filter, posX, posY)

    [hBuff, wBuff,~] = size(Buffer.buff);
    [hFilter, wFilter, tFilter] = size(Filter);

    % Iterate over filter
    for y = 1:hFilter
        % Compute position in buffer
        yBuff = y  - int32(hFilter/2) + posY;
        % skip if position is invalid
        if(yBuff < 1 || yBuff > hBuff)
            continue;
        end;
        for x = 1:wFilter
            xBuff = x - int32(wFilter/2) + posX;
            if(xBuff < 1 || xBuff > wBuff)
                continue;
            end;
            % For each time slice
            for t = 1:tFilter
                % Take value from filter and add to buffer
               % tBuff = mod(-t + Buffer.W - 2  + tFilter - 1,tFilter)+ 1;
                tBuff = mod(tFilter - t + Buffer.W , tFilter)+1;
                Buffer.buff(yBuff,xBuff,tBuff) = Buffer.buff(yBuff,xBuff,tBuff) + Filter(x,y,t);
            end
        end
    end
end

