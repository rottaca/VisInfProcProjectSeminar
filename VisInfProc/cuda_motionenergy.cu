
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_settings.h"
#include "cuda_helper.h"
#include "datatypes.h"
#include <assert.h>

#define MAX_SHARED_GPU_EVENTS 256
__global__ void kernelProcessEventsBatchAsync(SimpleEvent* gpuEventList,int gpuEventListSize,
                            float* gpuFilter, int fsx, int fsy, int fsz,
                            float* gpuBuffer, int ringBufferIdx,
                            int bsx, int bsy, int bsz,
                            int fs_xy, int fn){

    // Calculate filter idx
    int filterPos = threadIdx.x + blockIdx.x * blockDim.x;
    // Idx valid
    if (filterPos < fn){
        float filterVal = gpuFilter[filterPos];
        // Compute x,y,z coodinates in buffer
        int fz = filterPos / fs_xy;
        int fxy = filterPos % fs_xy;
        int fy = fxy / fsx;
        int fx = fxy % fsx;

        // Convert buffer z index (flip z)
        int bz = ((ringBufferIdx + (fsz - 1) - fz ) % bsz);
        int bx_tmp = fsx / 2 - fx;
        int by_tmp = fsy / 2 - fy;
        int bPos_tmp = bz*bsy*bsx;

        // Per block shared memory
        __shared__ SimpleEvent gpuEventListShared[MAX_SHARED_GPU_EVENTS];
        int eventGroupCnt = ceil(gpuEventListSize/(float)MAX_SHARED_GPU_EVENTS);
        // Load events blockwise
        for(int eventGroupIdx = 0; eventGroupIdx<eventGroupCnt; eventGroupIdx++){
            int globalEventIdx = eventGroupIdx*MAX_SHARED_GPU_EVENTS+threadIdx.x/2;
            // The first MAX_SHARED_GPU_EVENTS threads copy the event data into shared memory
            if(threadIdx.x/2 < MAX_SHARED_GPU_EVENTS && globalEventIdx < gpuEventListSize){
                // even threads load x, odd threads load y
                if(threadIdx.x % 2 == 0){
                    gpuEventListShared[threadIdx.x/2].x = gpuEventList[globalEventIdx].x;
                }else{
                    gpuEventListShared[threadIdx.x/2].y = gpuEventList[globalEventIdx].y;
                }
            }
            // Synchronize
            __syncthreads();

            // Iterate over every event block in shared memory
            for(int localEventIdx = 0; localEventIdx < MAX_SHARED_GPU_EVENTS &&
                eventGroupIdx*MAX_SHARED_GPU_EVENTS+localEventIdx < gpuEventListSize; localEventIdx++){
                // Compute corresponding buffer coordinate (flip filter x,y)
                int bx = bx_tmp + gpuEventListShared[localEventIdx].x;
                int by = by_tmp + gpuEventListShared[localEventIdx].y;

                // Check for valid buffer position (filp buffer z)
                if(bx >= 0 && bx < bsx && by >= 0 && by < bsy){
                    int bufferPos = bPos_tmp + by*bsx + bx;
                    atomicAdd(gpuBuffer + bufferPos,filterVal);
                }
            }
        }
    }
}

__host__ void cudaProcessEventsBatchAsync(SimpleEvent* gpuEventList,int gpuEventListSize,
                                          float* gpuFilter, int fsx, int fsy, int fsz,
                                          float* gpuBuffer, int ringBufferIdx,
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

__global__ void kernelReadOpponentMotionEnergyAsync(float* gpuConvBufferl1,
                                                    float* gpuConvBufferl2,
                                                    float* gpuConvBufferr1,
                                                    float* gpuConvBufferr2,
                                                    int ringBufferIdx,
                                                    int bsx, int bsy, int bsz, int n,
                                                    float alphaPNorm, float alphaQNorm, float betaNorm, float sigmaNorm,
                                                    float* gpuEnergyBuffer){
    int bufferPos = threadIdx.x + blockIdx.x * blockDim.x;
    if(bufferPos < n){
        int bx,by,bz;
        int bxy = bufferpos / (bsx*bsy);
        bz = bufferPos % bsx*bsy;
        bx = bxy % bsx;
        by = bxy / bsx;

        // Offset in ringbuffer
        int bufferPosConv = bufferPos + ringBufferIdx*bsx*bsy;
        // Get answer from all 4 corresponding buffers and compute opponent motion energy
        // get all four filter responses and reset buffers
        float l1 = gpuConvBufferl1[bufferPosConv];
        gpuConvBufferl1[bufferPosConv] = 0;
        float l2 = gpuConvBufferl2[bufferPosConv];
        gpuConvBufferl2[bufferPosConv] = 0;
        float r1 = gpuConvBufferr1[bufferPosConv];
        gpuConvBufferr1[bufferPosConv] = 0;
        float r2 = gpuConvBufferr2[bufferPosConv];
        gpuConvBufferr2[bufferPosConv] = 0;

        // Compute opponent motion energy
        float energyR = sqrt(r1*r1+r2*r2);
        float energyL = sqrt(l1*l1+l2*l2);

        // Normalize energy
//        q_i = 0;
//        for(int y = -1; y <= 1; y++){
//            int by_ = by + y;
//            if(by_ < 0 || by_ >= bsy)
//                continue;
//            for(int x = -1; x <= 1; x++){
//                int bx_ = bx + x;
//                if(bx_ < 0 || bx_ >= bsx)
//                    continue;

//            }
//        }

        gpuEnergyBuffer[bufferPos] = energyR - energyL;
    }
}

__host__ void cudaReadOpponentMotionEnergyAsync(float* gpuConvBufferl1,
                                                float* gpuConvBufferl2,
                                                float* gpuConvBufferr1,
                                                float* gpuConvBufferr2,
                                                int ringBufferIdx,
                                                int bsx, int bsy, int bsz,
                                                float alphaPNorm, float alphaQNorm, float betaNorm, float sigmaNorm,
                                                float* gpuEnergyBuffer,
                                                cudaStream_t cudaStream)
{
    int n = bsx*bsy;
    long blocks = ceil((float)n/THREADS_PER_BLOCK);
    kernelReadOpponentMotionEnergyAsync<<<blocks,THREADS_PER_BLOCK,0,cudaStream>>>(gpuConvBufferl1,
                                                                                   gpuConvBufferl2,
                                                                                   gpuConvBufferr1,
                                                                                   gpuConvBufferr2,
                                                                                   ringBufferIdx,bsx,bsy,bsz,n,
                                                                                   alphaPNorm,alphaQNorm,betaNorm,sigmaNorm,
                                                                                   gpuEnergyBuffer);
}
