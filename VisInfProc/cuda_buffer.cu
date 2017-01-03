

#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_helper.h"
#include "cuda_settings.h"


__global__ void kernel2DBufferToRGBImage(int sx, int sy, int s, double min, double max,
                                         double* gpuBuffer, unsigned char* gpuImage){

    int buffIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if(buffIdx < s){
        double v = gpuBuffer[buffIdx];

        int colorIdx = round((v - min)/(max-min)*(GPU_LUT_COLORMAP_SZ-1));
        if(colorIdx < 0)
            colorIdx = 0;
        else if(colorIdx >= GPU_LUT_COLORMAP_SZ)
            colorIdx = GPU_LUT_COLORMAP_SZ-1;

        colorIdx *= 3;
        buffIdx *= 3;
        gpuImage[buffIdx++] = GPUrgbColormapLUT[colorIdx++];
        gpuImage[buffIdx++] = GPUrgbColormapLUT[colorIdx++];
        gpuImage[buffIdx] = GPUrgbColormapLUT[colorIdx];
    }
}

__host__ void cuda2DBufferToRGBImage(int sx, int sy,double min, double max,
                               double* gpuBuffer, unsigned char* gpuImage){
    int s = sx*sy;
    long blocks = ceil((float)(s)/THREADS_PER_BLOCK);
    kernel2DBufferToRGBImage<<<blocks,THREADS_PER_BLOCK>>>(sx,sy,s,min, max,gpuBuffer,gpuImage);
}
