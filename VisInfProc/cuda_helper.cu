#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "cuda_settings.h"
#include "cuda_helper.h"

#include <iostream>

__host__ void* cudaCreateBuffer(long size){
    if(size == 0)
        return NULL;
    void* gpuPtr;
    gpuErrchk(cudaMalloc(&gpuPtr,size));
    if(gpuPtr == 0)
    {
        fprintf(stderr,"Cuda malloc failed! Can't create buffer of size: %d\n",size);
        exit(-1);
    }
    return gpuPtr;
}
__host__ void cudaUploadBuffer(void* cpuBuffPtr, void* gpuBuffPtr,long size,cudaStream_t stream){
    gpuErrchk(cudaMemcpyAsync(gpuBuffPtr,cpuBuffPtr,size,cudaMemcpyHostToDevice,stream));
}
__host__ void cudaDownloadBuffer(void* gpuBuffPtr, void * cpuBuffPtr,long size,cudaStream_t stream){
    gpuErrchk(cudaMemcpyAsync(cpuBuffPtr,gpuBuffPtr,size,cudaMemcpyDeviceToHost,stream));
}
__host__ void cudaCopyBuffer(void* gpuBuffPtrDest, void * gpuBuffPtrSrc,long size,cudaStream_t stream){
    gpuErrchk(cudaMemcpyAsync(gpuBuffPtrDest,gpuBuffPtrSrc,size,cudaMemcpyDeviceToDevice,stream));
}
__host__ void cudaFreeBuffer(void* gpuBuffPtr){
    gpuErrchk(cudaFree(gpuBuffPtr));
}

__global__ void kernelSetDoubleBuffer(float* gpuBuffPtr, float v, long size){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size)
        gpuBuffPtr[index] = v;
}

__host__ void cudaSetDoubleBuffer(float* gpuBuffPtr,float v, long size,cudaStream_t stream){
    // Run through filter buffer
    long blocks = ceil((float)size/THREADS_PER_BLOCK);
    kernelSetDoubleBuffer<<<blocks,THREADS_PER_BLOCK,0,stream>>>(gpuBuffPtr,v,size);
}
