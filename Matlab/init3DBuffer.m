function [ Buffer ] = init3DBuffer( w,h,t )
    Buffer= {};
    
    Buffer.buff = zeros(h,w,t);
    
    Buffer.R = 1;
    Buffer.W = 1;
end