
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_settings.h"


__global__ void kernelCombineFilters(int sx, int sy, int sz,
                                     double* gpuTemp, double* gpuSpatial,
                                     double* gpuCombined){
    int combinedIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int s_xy = sx*sy;
    int z = combinedIdx / s_xy;
    int xy = combinedIdx % s_xy;

    gpuCombined[combinedIdx] = gpuTemp[z]*gpuSpatial[xy];
}

__host__ void cudaCombineFilters(int sx, int sy, int sz,
                                 double* gpuTemp, double* gpuSpatial,
                                 double* gpuCombined)
{

    long blocks = ceil((float)(sx*sy*sz)/THREADS_PER_BLOCK);
    kernelCombineFilters<<<blocks,THREADS_PER_BLOCK>>>(
                         sx,sy,sz,gpuTemp,gpuSpatial,gpuCombined);
}
