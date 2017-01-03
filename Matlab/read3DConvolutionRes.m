function [ Buffer, res ] = read3DConvolutionRes( Buffer )
    res = Buffer.buff(:,:,Buffer.W+1);
    Buffer.buff(:,:,Buffer.W+1) = zeros(size(Buffer.buff(:,:,Buffer.W+1)));
    Buffer.W = mod(Buffer.W+1,size(Buffer.buff,3));
    
end
