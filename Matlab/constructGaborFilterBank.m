function [ result ] = constructGaborFilterBank( theta, f0, angles, halfGaborRange )
    
    result = {};
    for i=angles
        angle = i/360*2*pi();
        % Compute fx and fy
        fx0 = cos(angle)*f0;
        fy0 = sin(angle)*f0;
        result(end+1).fx0 = fx0;
        result(end).fy0 = fy0;
        result(end).angle = angle;
        
        [Godd, Geven] = constructGaborFilter(theta, fx0, fy0, halfGaborRange);
        result(end).Go = Godd;
        result(end).Ge = Geven;
    end
end