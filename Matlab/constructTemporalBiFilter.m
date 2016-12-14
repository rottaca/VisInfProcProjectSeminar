function [ res ] = constructTemporalBiFilter( sigma1, u1, s1, sigma2, u2, s2, times )

    G = @(theta,u,t)exp(-(t-u).^2/(2*theta^2));
    
    res = -s1*G(sigma1,u1,times) + s2*G(sigma2,u2,times);
end

