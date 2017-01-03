function [ Buffer ] = init3DBuffer( w,h,t )
    Buffer= {};
    
    Buffer.buff = zeros(h,w,t);
    
    Buffer.W = 0;
end