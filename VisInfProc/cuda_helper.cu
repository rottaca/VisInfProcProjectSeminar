#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "cuda_settings.h"
#include "cuda_helper.h"

#include <iostream>
/**
 * @brief cudaCreateBuffer Allocates a cuda buffer and stops the programm on error.
 * @param size
 * @return
 */
__host__ void* cudaCreateBuffer(size_t size)
{
    if(size == 0)
        return NULL;
    void* gpuPtr;
    gpuErrchk(cudaMalloc(&gpuPtr,size));
    if(gpuPtr == 0) {
        fprintf(stderr,"Cuda malloc failed! Can't create buffer of size: %ld\n",size);
        exit(-1);
    }
    return gpuPtr;
}
/**
 * @brief cudaUploadBuffer Uploads a cpu buffer in a gpu buffer and does error handling.
 * @param cpuBuffPtr
 * @param gpuBuffPtr
 * @param size
 * @param stream
 */
__host__ void cudaUploadBuffer(void* cpuBuffPtr, void* gpuBuffPtr, size_t size, cudaStream_t stream)
{
    gpuErrchk(cudaMemcpyAsync(gpuBuffPtr,cpuBuffPtr,size,cudaMemcpyHostToDevice,stream));
}
/**
 * @brief cudaDownloadBuffer Downloads a gpu buffer into a cpu buffer and does error handling.
 * @param gpuBuffPtr
 * @param cpuBuffPtr
 * @param size
 * @param stream
 */
__host__ void cudaDownloadBuffer(void* gpuBuffPtr, void * cpuBuffPtr,size_t size,cudaStream_t stream)
{
    gpuErrchk(cudaMemcpyAsync(cpuBuffPtr,gpuBuffPtr,size,cudaMemcpyDeviceToHost,stream));
}
/**
 * @brief cudaFreeBuffer Copies a gpu buffer to an other gpu buffer and does error handling.
 * @param gpuBuffPtr
 */
__host__ void cudaCopyBuffer(void* gpuBuffPtrDest, void * gpuBuffPtrSrc, size_t size, cudaStream_t stream)
{
    gpuErrchk(cudaMemcpyAsync(gpuBuffPtrDest,gpuBuffPtrSrc,size,cudaMemcpyDeviceToDevice,stream));
}
/**
 * @brief cudaFreeBuffer Releases a gpu buffer and does error handling.
 * @param gpuBuffPtr
 */
__host__ void cudaFreeBuffer(void* gpuBuffPtr)
{
    gpuErrchk(cudaFree(gpuBuffPtr));
}

__global__ void kernelSetDoubleBuffer(float* gpuBuffPtr, float v, size_t size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size)
        gpuBuffPtr[index] = v;
}
/**
 * @brief cudaSetDoubleBuffer Sets a gpu double buffer to a value v.
 * @param gpuBuffPtr
 * @param v
 * @param size
 * @param stream
 */
__host__ void cudaSetDoubleBuffer(float* gpuBuffPtr,float v, size_t size,cudaStream_t stream)
{
    // Run through filter buffer
    size_t blocks = ceil((float)size/THREADS_PER_BLOCK);
    kernelSetDoubleBuffer<<<blocks,THREADS_PER_BLOCK,0,stream>>>(gpuBuffPtr,v,size);
}
