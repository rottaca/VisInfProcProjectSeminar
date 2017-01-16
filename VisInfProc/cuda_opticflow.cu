
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_settings.h"
#include "cuda_helper.h"


__global__ void kernelComputeOpticFlow(int n,
                                       float* gpuFlowX,float* gpuFlowY,
                                       float** gpuEnergy,float* orientations, int orientationCnt){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n){
        gpuFlowX[idx] = 0;
        gpuFlowY[idx] = 0;
        for(int i = 0; i  < orientationCnt; i++)
        {
            float energy = gpuEnergy[i][idx];
            gpuFlowX[idx] += energy*cos(orientations[i]);
            gpuFlowY[idx] += energy*sin(orientations[i]);
        }
    }
}

__host__ void cudaComputeOpticFlow(int sx, int sy,
                                  float* gpuFlowX, float* gpuFlowY,
                                  float** gpuArrGpuEnergy, float* gpuArrOrientations, int orientationCnt, cudaStream_t stream)
{
    int n = sx*sy;
    long blocks = ceil((float)n/THREADS_PER_BLOCK);
    kernelComputeOpticFlow<<<blocks,THREADS_PER_BLOCK,0,stream>>>(
                         n,
                         gpuFlowX,gpuFlowY,
                         gpuArrGpuEnergy,gpuArrOrientations,orientationCnt);
}
