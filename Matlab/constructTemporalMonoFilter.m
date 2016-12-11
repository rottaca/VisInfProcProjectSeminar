function [ res ] = constructTemporalMonoFilter( sigma,u,times )

    G= @(t)exp(-(t-u).^2/(2*sigma^2));
       
    res = G(times);
end

