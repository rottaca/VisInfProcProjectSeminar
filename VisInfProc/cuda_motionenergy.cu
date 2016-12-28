
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_settings.h"


__global__ void kernelComputeOpponentMotionEnergy(int sx, int sy,int n,
                                                  double* gpul1,double* gpul2,
                                                  double* gpur1,double* gpur2,
                                                 double* gpuEnergy){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n)
        gpuEnergy[idx] = sqrt(gpur1[idx]*gpur1[idx] + gpur2[idx]*gpur2[idx])
                -sqrt(gpul1[idx]*gpul1[idx] + gpul2[idx]*gpul2[idx]);
}

__host__ void cudaComputeOpponentMotionEnergy(int sx, int sy,
                                  double* gpul1,double* gpul2,
                                  double* gpur1,double* gpur2,
                                 double* gpuEnergy)
{
    int n = sx*sy;
    long blocks = ceil((float)n/THREADS_PER_BLOCK);
    kernelComputeOpponentMotionEnergy<<<blocks,THREADS_PER_BLOCK>>>(
                         sx,sy,n,gpul1,gpul2,gpur1,gpur2,gpuEnergy);
}
