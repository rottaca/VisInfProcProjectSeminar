function [ res ] = constructTemporalBiFilter( theta1, u1, s1, theta2, u2, s2, times )

    G = @(theta,u,t)exp(-(t-u).^2/(2*theta^2));
    
    res = -s1*G(theta1,u1,times) + s2*G(theta2,u2,times);
end

