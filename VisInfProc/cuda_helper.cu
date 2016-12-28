#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "cuda_settings.h"
#include "cuda_helper.h"

#include <iostream>

__host__ void* cudaCreateBuffer(long size){
    void* gpuPtr;
    gpuErrchk(cudaMalloc(&gpuPtr,size));
    if(gpuPtr == 0)
    {
        fprintf(stderr,"Cuda malloc failed!\n");
        exit(-1);
    }
    return gpuPtr;
}
__host__ void cudaUploadBuffer(void* cpuBuffPtr, void* gpuBuffPtr,long size){
    gpuErrchk(cudaMemcpy(gpuBuffPtr,cpuBuffPtr,size,cudaMemcpyHostToDevice));
}
__host__ void cudaDownloadBuffer(void* gpuBuffPtr, void * cpuBuffPtr,long size){
    gpuErrchk(cudaMemcpy(cpuBuffPtr,gpuBuffPtr,size,cudaMemcpyDeviceToHost));
}
__host__ void cudaCopyBuffer(void* gpuBuffPtrDest, void * gpuBuffPtrSrc,long size){
    gpuErrchk(cudaMemcpy(gpuBuffPtrDest,gpuBuffPtrSrc,size,cudaMemcpyDeviceToDevice));
}
__host__ void cudaFreeBuffer(void* gpuBuffPtr){
    gpuErrchk(cudaFree(gpuBuffPtr));
}

__global__ void kernelSetDoubleBuffer(double* gpuBuffPtr, double v, long size){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size)
        gpuBuffPtr[index] = v;
}

__host__ void cudaSetDoubleBuffer(double* gpuBuffPtr,double v, long size){
    // Run through filter buffer
    long blocks = ceil((float)size/THREADS_PER_BLOCK);
    kernelSetDoubleBuffer<<<blocks,THREADS_PER_BLOCK>>>(gpuBuffPtr,v,size);
}
