function [ Buffer, res ] = read3DConvolutionRes( Buffer )
    res = Buffer.buff(:,:,Buffer.R);
    Buffer.buff(:,:,Buffer.R) = zeros(size(Buffer.buff(:,:,Buffer.R)));
    Buffer.R = mod(Buffer.R,size(Buffer.buff,3))+1;
    
end
