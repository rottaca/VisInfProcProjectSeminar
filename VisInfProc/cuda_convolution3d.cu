
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include "cuda_helper.h"
#include "cuda_settings.h"
#include <math.h>

__global__ void kernelConvolute3D(double* gpuBuffer,
                                  int writeIdx, int bsx, int bsy, int bsz,
                                  double* gpuFilter, int fsx, int fsy, int fsz,
                                  int px, int py){

    int filterIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int fs_xy = fsx*fsy;

    if (filterIdx < fs_xy*fsz){
        int fz = filterIdx / fs_xy;
        int fxy = filterIdx % fs_xy;
        int fy = fxy / fsx;
        int fx = fxy % fsx;

        int bx = fsx / 2 - fx + px;
        int by = fsy / 2 - fy + py;

        // Valid buffer position
        if(bx >= 0 && bx < bsx && by >= 0 && by < bsy){
            int bz = ((writeIdx + (fsz - 1) - fz ) % bsz);

            int bufferIdx = bz*bsy*bsx + by*bsx + bx;
            gpuBuffer[bufferIdx] += gpuFilter[filterIdx];
            //gpuBuffer[bufferIdx] = fz*100/fsz;
        }
    }
}

__host__ void cudaConvolution3D(double* gpuBuffer,
                              int writeIdx, int bsx, int bsy, int bsz,
                              double* gpuFilter, int fsx, int fsy, int fsz,
                                int px, int py){
    // Run through filter buffer
    long blocks = ceil((float)(fsx*fsy*fsz)/THREADS_PER_BLOCK);
    kernelConvolute3D<<<blocks,THREADS_PER_BLOCK>>>(gpuBuffer,writeIdx,bsx,bsy,bsz,
                                                    gpuFilter,fsx,fsy,fsz,
                                                    px,py);
}

