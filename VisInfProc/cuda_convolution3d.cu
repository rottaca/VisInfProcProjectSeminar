#ifndef _CU_CONVOLUTE3D_
#define _CU_CONVOLUTE3D_

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "cuda_helper.h"
#include "cuda_settings.h"

__global__ void kernelConvolute3D(double* gpuBuffer,
                                  int writeIdx, int bsx, int bsy, int bsz,
                                  double* gpuFilter, int fsx, int fsy, int fsz,
                                  int px, int py){

    int filterIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (filterIdx < fsx*fsy*fsz){
        // Compute buffer index and check bounds TODO
        int bufferIdx ;
    }

}

__host__ void cudaConvolution3D(double* gpuBuffer,
                              int writeIdx, int bsx, int bsy, int bsz,
                              double* gpuFilter, int fsx, int fsy, int fsz,
                                int px, int py){
    // Run through filter buffer
    long blocks = (fsx*fsy*fsz)/THREADS_PER_BLOCK;

    kernelConvolute3D<<<blocks,THREADS_PER_BLOCK>>>(gpuBuffer,writeIdx,bsx,bsy,bsz,
                                                    gpuFilter,fsx,fsy,fsz,
                                                    px,py);
    gpuKernelErrorCheck();
}


#endif
