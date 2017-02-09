
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_settings.h"
#include "cuda_helper.h"
#include "datatypes.h"
#include <assert.h>

__global__ void kernelProcessEventsBatchAsync(uint8_t* gpuEventsX,uint8_t* gpuEventsY,int gpuEventListSize,
        float* gpuFilter, int fsx, int fsy, int fsz,
        float* gpuBuffer, int ringBufferIdx,
        int bsx, int bsy, int bsz,
        int fs_xy, int fn)
{

    // Calculate filter idx
    int filterPos = threadIdx.x + blockIdx.x * blockDim.x;

    // Per block shared memory
    __shared__ uint8_t gpuEventListSharedX[MAX_SHARED_GPU_EVENTS];
    __shared__ uint8_t gpuEventListSharedY[MAX_SHARED_GPU_EVENTS];
    // How many runs do we need to process all events
    int processingRuns = ceilf((float)gpuEventListSize/MAX_SHARED_GPU_EVENTS);
    // Events for each thread to read
    int eventReadsPerThread = ceilf((float)MAX_SHARED_GPU_EVENTS/blockDim.x);
    // Offset n global event buffer
    int globalEventIdx = threadIdx.x;

    // Idx valid
    if (filterPos < fn) {
        // Read filter coefficient from global memory
        float filterVal = gpuFilter[filterPos];
        // Compute x,y,z coodinates in buffer
        int fz = filterPos / fs_xy;
        int fxy = filterPos % fs_xy;
        int fy = fxy / fsx;
        int fx = fxy % fsx;

        // Convert buffer coordinates (mirror all axes -> convolution instead of correlation)
        // Origin for mirroring is x = w/2, y = h/2, z = 0
        int bz = ((ringBufferIdx + (fsz - 1) - fz ) % bsz);
        int bx_tmp = fsx / 2 - fx;
        int by_tmp = fsy / 2 - fy;
        int bPos_tmp = bz*bsy*bsx;

        int sharedEventCnt = MAX_SHARED_GPU_EVENTS;
        // Iterate over event list in blocks, stored in shared memory
        for(int runIdx = 0; runIdx<processingRuns; runIdx++) {
            // Last run ? Compute size of shared event list
            if(runIdx+1 == processingRuns) {
                sharedEventCnt = gpuEventListSize % MAX_SHARED_GPU_EVENTS;
            }
            // Compute index in shared memory
            int localEventIdx = threadIdx.x;

            // Fill the shared memory either with MAX_SHARED_GPU_EVENTS
            // or use each thread mutlible times
            for(int i = 0; i < eventReadsPerThread; i++) {
                // Valid indices
                if(localEventIdx >= sharedEventCnt)
                    break;
                // Load event into shared memory by using one thread per event
                gpuEventListSharedX[localEventIdx] = gpuEventsX[globalEventIdx];
                gpuEventListSharedY[localEventIdx] = gpuEventsY[globalEventIdx];

                // Goto next event for which this thread is responsible
                localEventIdx += blockDim.x;
                globalEventIdx += blockDim.x;
            }

            // Synchronize threads and wait until shared memory is filled
            // TODO: Deadlock possible?
            // At least one thread in each warp should hit that barrier to continue!
            // Bad relationship between shared event list size and block size could cause problems ?!
            __syncthreads();

            // Iterate over every event block in shared memory
            for(localEventIdx = 0; localEventIdx < sharedEventCnt; localEventIdx++) {
                // Compute corresponding buffer coordinate
                int bx = bx_tmp + gpuEventListSharedX[localEventIdx];
                int by = by_tmp + gpuEventListSharedY[localEventIdx];

                // Check for valid buffer position (filp buffer z)
                if(bx >= 0 && bx < bsx && by >= 0 && by < bsy) {
                    int bufferPos = bPos_tmp + by*bsx + bx;
                    // Add each filter coefficient to the global buffer
                    atomicAdd(gpuBuffer + bufferPos,filterVal);
                }
            }
        }
    }
}
/**
 * @brief cudaProcessEventsBatchAsync Processes a gpu event list with a given
 *                                    filter and stores the result in the given buffer
 * @param gpuEventsX
 * @param gpuEventsY
 * @param gpuEventListSize
 * @param gpuFilter
 * @param fsx
 * @param fsy
 * @param fsz
 * @param gpuBuffer
 * @param ringBufferIdx
 * @param bsx
 * @param bsy
 * @param bsz
 * @param cudaStream
 */
__host__ void cudaProcessEventsBatchAsync(uint8_t* gpuEventsX,uint8_t* gpuEventsY,int gpuEventListSize,
        float* gpuFilter, int fsx, int fsy, int fsz,
        float* gpuBuffer, int ringBufferIdx,
        int bsx, int bsy, int bsz,
        cudaStream_t cudaStream)
{
    int fs_xy = fsx*fsy;
    int fn = fs_xy*fsz;
    size_t blocks = ceilf((float)fn/THREADS_PER_BLOCK);
    kernelProcessEventsBatchAsync<<<blocks,THREADS_PER_BLOCK,0,cudaStream>>>(gpuEventsX,gpuEventsY,gpuEventListSize,
            gpuFilter,fsx,fsy,fsz,
            gpuBuffer,ringBufferIdx,
            bsx,bsy,bsz,
            fs_xy,fn);
}

__global__ void kernelReadMotionEnergyAsync(float* gpuConvBufferl1,
        float* gpuConvBufferl2,
        int ringBufferIdx,
        int bsx, int bsy, int n,
        float* gpuEnergyBuffer)
{
    int bufferPos = threadIdx.x + blockIdx.x * blockDim.x;
    if(bufferPos < n) {
        // Offset in ringbuffer
        int bufferPosConv = bufferPos + ringBufferIdx*bsx*bsy;
        // Get answer from two corresponding buffers and compute motion energy
        float l1 = gpuConvBufferl1[bufferPosConv];
        float l2 = gpuConvBufferl2[bufferPosConv];

        // Compute motion energy
        gpuEnergyBuffer[bufferPos] = sqrt(l1*l1+l2*l2);
    }
}
/**
 * @brief cudaReadMotionEnergyAsync Reads the motionenergy from the two
 *                                  corresponding convolution buffers
 *                                  and stores the energy in a gpu buffer.
 * @param gpuConvBufferl1
 * @param gpuConvBufferl2
 * @param ringBufferIdx
 * @param bsx
 * @param bsy
 * @param gpuEnergyBuffer
 * @param cudaStream
 */
__host__ void cudaReadMotionEnergyAsync(float* gpuConvBufferl1,
                                        float* gpuConvBufferl2,
                                        int ringBufferIdx,
                                        int bsx, int bsy,
                                        float* gpuEnergyBuffer,
                                        cudaStream_t cudaStream)
{
    int n = bsx*bsy;
    size_t blocks = ceilf((float)n/THREADS_PER_BLOCK);
    kernelReadMotionEnergyAsync<<<blocks,THREADS_PER_BLOCK,0,cudaStream>>>(gpuConvBufferl1,
            gpuConvBufferl2,
            ringBufferIdx,bsx,bsy,n,
            gpuEnergyBuffer);
}


__global__ void kernelNormalizeMotionEnergyAsync(int bsx, int bsy, int n,
        float alphaPNorm, float alphaQNorm, float betaNorm, float sigmaNorm,
        float* gpuEnergyBuffer)
{
    int bufferPos = threadIdx.x + blockIdx.x * blockDim.x;
    float sigmaNorm2_2 = 2*sigmaNorm*sigmaNorm;
    if(bufferPos < n) {
        int bx,by;
        int bxy = bufferPos / (bsx*bsy);
        bx = bxy % bsx;
        by = bxy / bsx;
        // Read energy
        float I = gpuEnergyBuffer[bufferPos];
        float q_i = 0;
        // Normalize over 5x5 region
        for(int y = -2; y <= 2; y++) {
            int by_ = by + y;

            if(by_ < 0 || by_ >= bsy)
                continue;

            for(int x = -2; x <= 2; x++) {
                int bx_ = bx + x;

                if(bx_ < 0 || bx_ >= bsx ||
                        (bx == bx_ && by == by_))
                    continue;
                // TODO
                // Each thread computes the same
                float gaus = 1/(sigmaNorm2_2*M_PI)* exp(-(bx_*bx_ + by_*by_)/sigmaNorm2_2);
                // TODO Use shared memory to avoid extra global memory access
                q_i += gpuEnergyBuffer[by_*bsx+bx_]*gaus;
            }
        }
        q_i /= alphaQNorm;

        // Compute p_i
        float p_i = (I*betaNorm)/(alphaPNorm + I + q_i);

        // Use normalized value
        gpuEnergyBuffer[bufferPos] = p_i;
    }
}
/**
 * @brief cudaNormalizeMotionEnergyAsync Normalizes the motion energy
 *                                       inplace.
 * @param bsx
 * @param bsy
 * @param alphaPNorm
 * @param alphaQNorm
 * @param betaNorm
 * @param sigmaNorm
 * @param gpuEnergyBuffer
 * @param cudaStream
 */
__host__ void cudaNormalizeMotionEnergyAsync(int bsx, int bsy,
        float alphaPNorm, float alphaQNorm, float betaNorm, float sigmaNorm,
        float* gpuEnergyBuffer,
        cudaStream_t cudaStream)
{
    int n = bsx*bsy;
    size_t blocks = ceilf((float)n/THREADS_PER_BLOCK);
    kernelNormalizeMotionEnergyAsync<<<blocks,THREADS_PER_BLOCK,0,cudaStream>>>(bsx,bsy,n,
            alphaPNorm,alphaQNorm,betaNorm,sigmaNorm,
            gpuEnergyBuffer);
}
