function [Godd, Geven, X, Y] = constructGaborFilter(theta, fx0, fy0, halfGaborRange)
    tmp = 2*pi()/theta^2;
    
    p = -halfGaborRange:halfGaborRange;
    [X,Y] = meshgrid(p,p);
    
    gauss = exp(-2*pi^2.*(X.^2+Y.^2)./theta^2);
    
    real = cos(2*pi.*(fx0.*X + fy0.*Y));
    imag = sin(2*pi.*(fx0.*X + fy0.*Y));
    
    Godd = tmp*imag.*gauss;
    Geven = tmp*real.*gauss;
end