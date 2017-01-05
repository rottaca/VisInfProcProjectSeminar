
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_settings.h"
#include "cuda_helper.h"
#include "datatypes.h"
#include <assert.h>

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


__global__ void kernelProcessEventsBatchAsync(SimpleEvent* gpuEventList,int gpuEventListSize,
                            double* gpuFilter, int fsx, int fsy, int fsz,
                            double* gpuBuffer, int ringBufferIdx,
                            int bsx, int bsy, int bsz,
                            int fs_xy, int fn){

    // Calculate filter idx
    int filterPos = threadIdx.x + blockIdx.x * blockDim.x;
    // Idx valid
    if (filterPos < fn){
        // Compute x,y,z coodinates in buffer
        int fz = filterPos / fs_xy;
        int fxy = filterPos % fs_xy;
        int fy = fxy / fsx;
        int fx = fxy % fsx;

        // Iterate over every event in block
        for(int eventIdx = 0; eventIdx < gpuEventListSize; eventIdx++){
            // Compute corresponding buffer coordinate (flip filter x,y)
            int bx = fsx / 2 - fx + gpuEventList[eventIdx].x;
            int by = fsy / 2 - fy + gpuEventList[eventIdx].y;

            // Check for valid buffer position (filp buffer z)
            if(bx >= 0 && bx < bsx && by >= 0 && by < bsy){

                // Convert buffer z index (flip z)
                int bz = ((ringBufferIdx + (fsz - 1) - fz ) % bsz);
                int bufferPos = bz*bsy*bsx + by*bsx + bx;

                gpuBuffer[bufferPos] += gpuFilter[filterPos];
            }
        }
    }
}

__host__ void cudaProcessEventsBatchAsync(SimpleEvent* gpuEventList,int gpuEventListSize,
                                          double* gpuFilter, int fsx, int fsy, int fsz,
                                          double* gpuBuffer, int ringBufferIdx,
                                          int bsx, int bsy, int bsz,
                                          cudaStream_t cudaStream)
{
    int fs_xy = fsx*fsy;
    int fn = fs_xy*fsz;
    long blocks = ceil((float)fn/THREADS_PER_BLOCK);
    kernelProcessEventsBatchAsync<<<blocks,THREADS_PER_BLOCK,0,cudaStream>>>(gpuEventList,gpuEventListSize,
                                                                             gpuFilter,fsx,fsy,fsz,
                                                                             gpuBuffer,ringBufferIdx,
                                                                             bsx,bsy,bsz,
                                                                             fs_xy,fn);
}

__global__ void kernelReadOpponentMotionEnergyAsync(double* gpuConvBufferl1,
                                                    double* gpuConvBufferl2,
                                                    double* gpuConvBufferr1,
                                                    double* gpuConvBufferr2,
                                                    int ringBufferIdx,
                                                    int bsx, int bsy, int bsz, int n,
                                                    double* gpuEnergyBuffer){
    int bufferPos = threadIdx.x + blockIdx.x * blockDim.x;
    if(bufferPos < n){
        // Offset in ringbuffer
        int bufferPosConv = bufferPos + ringBufferIdx*bsx*bsy;
        // Get answer from all 4 corresponding buffers and compute opponent motion energy
        // get all four filter responses and reset buffers
        double l1 = gpuConvBufferl1[bufferPosConv];
        gpuConvBufferl1[bufferPosConv] = 0;
        double l2 = gpuConvBufferl2[bufferPosConv];
        gpuConvBufferl2[bufferPosConv] = 0;
        double r1 = gpuConvBufferr1[bufferPosConv];
        gpuConvBufferr1[bufferPosConv] = 0;
        double r2 = gpuConvBufferr2[bufferPosConv];
        gpuConvBufferr2[bufferPosConv] = 0;

        // Compute opponent motion energy
        gpuEnergyBuffer[bufferPos] = sqrt(r1*r1+r2*r2) - sqrt(l1*l1+l2*l2);
    }
}

__host__ void cudaReadOpponentMotionEnergyAsync(double* gpuConvBufferl1,
                                                double* gpuConvBufferl2,
                                                double* gpuConvBufferr1,
                                                double* gpuConvBufferr2,
                                                int ringBufferIdx,
                                                int bsx, int bsy, int bsz,
                                                double* gpuEnergyBuffer,
                                                cudaStream_t cudaStream)
{
    int n = bsx*bsy;
    long blocks = ceil((float)n/THREADS_PER_BLOCK);
    kernelReadOpponentMotionEnergyAsync<<<blocks,THREADS_PER_BLOCK,0,cudaStream>>>(gpuConvBufferl1,
                                                                                   gpuConvBufferl2,
                                                                                   gpuConvBufferr1,
                                                                                   gpuConvBufferr2,
                                                                                    ringBufferIdx,bsx,bsy,bsz,n,
                                                                                    gpuEnergyBuffer);
}
