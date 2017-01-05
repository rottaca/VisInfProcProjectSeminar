
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
                            double** gpuFilters, int fsx, int fsy, int fsz,
                            double** gpuBuffers, int ringBufferIdx,
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
                for(int bufferIdx = 0;  bufferIdx < 4; bufferIdx++){
                        gpuBuffers[bufferIdx][bufferPos] += gpuFilters[bufferIdx][filterPos];
                }
            }
        }
    }
}

__host__ void cudaProcessEventsBatchAsync(SimpleEvent* gpuEventList,int gpuEventListSize,
                                          double** gpuFilters, int fsx, int fsy, int fsz,
                                          double** gpuBuffers, int ringBufferIdx,
                                          int bsx, int bsy, int bsz,
                                          cudaStream_t cudaStream)
{
    int fs_xy = fsx*fsy;
    int fn = fs_xy*fsz;
    long blocks = ceil((float)fn/THREADS_PER_BLOCK);
    kernelProcessEventsBatchAsync<<<blocks,THREADS_PER_BLOCK,0,cudaStream>>>(gpuEventList,gpuEventListSize,
                                                                             gpuFilters,fsx,fsy,fsz,
                                                                             gpuBuffers,ringBufferIdx,
                                                                             bsx,bsy,bsz,
                                                                             fs_xy,fn);
}

__global__ void kernelReadOpponentMotionEnergyAsync(double** gpuConvBuffers,int bufferFilterCount,int ringBufferIdx,
                                                    int bsx, int bsy, int bsz, int n,
                                                    double** gpuEnergyBuffers, int cnt){
    int bufferPos = threadIdx.x + blockIdx.x * blockDim.x;
    if(bufferPos < n){
        // Offset in ringbuffer
        int bufferPosConv = bufferPos + ringBufferIdx*bsx*bsy;
        // Get answer from all 4 corresponding buffers and compute opponent motion energy
        int localI = 0;
        for(int bufferIdx = 0; bufferIdx < cnt; bufferIdx++){
            // get all four filter responses and reset buffers
            double l1 = gpuConvBuffers[localI][bufferPosConv];
            gpuConvBuffers[localI][bufferPosConv] = 0;
            localI++;
            double l2 = gpuConvBuffers[localI][bufferPosConv];
            gpuConvBuffers[localI][bufferPosConv] = 0;
            localI++;
            double r1 = gpuConvBuffers[localI][bufferPosConv];
            gpuConvBuffers[localI][bufferPosConv] = 0;
            localI++;
            double r2 = gpuConvBuffers[localI][bufferPosConv];
            gpuConvBuffers[localI][bufferPosConv] = 0;
            localI++;

            // Compute opponent motion energy
            gpuEnergyBuffers[bufferIdx][bufferPos] = sqrt(r1*r1+r2*r2) - sqrt(l1*l1+l2*l2);
        }
    }
}

__host__ void cudaReadOpponentMotionEnergyAsync(double** gpuConvBuffers,int bufferFilterCount,int ringBufferIdx,
                                                int bsx, int bsy, int bsz,
                                           double** gpuEnergyBuffers, int cnt,
                                           cudaStream_t cudaStream)
{
    int n = bsx*bsy;
    long blocks = ceil((float)n/THREADS_PER_BLOCK);
    kernelReadOpponentMotionEnergyAsync<<<blocks,THREADS_PER_BLOCK,0,cudaStream>>>(gpuConvBuffers,bufferFilterCount,
                                                                             ringBufferIdx,bsx,bsy,bsz,n,
                                                                             gpuEnergyBuffers,cnt);
}
