function [ res ] = constructTemporalMonoFilter( theta,u,times )

    G= @(t)exp(-(t-u).^2/(2*theta^2));
       
    res = G(times);
end

