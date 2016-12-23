#ifndef _CU_HELPER_
#define _CU_HELPER_

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "cuda_settings.h"
#include "cuda_helper.h"



__host__ double* cudaCreateBuffer(long size){
    double* gpuPtr;
    gpuErrchk(cudaMalloc(&gpuPtr,size*sizeof(double)));
    if(gpuPtr == 0)
    {
        fprintf(stderr,"Cuda malloc failed!\n");
        exit(-1);
    }
    return gpuPtr;
}

__host__ void cudaUploadBuffer(double* cpuBuffPtr, double * gpuBuffPtr,long size){
    gpuErrchk(cudaMemcpy(gpuBuffPtr,cpuBuffPtr,size*sizeof(double),cudaMemcpyHostToDevice));
}

__global__ void kernelSetBuffer(double* gpuBuffPtr, double v, long size){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size)
        gpuBuffPtr[index] = v;
}

__host__ void cudaSetBuffer(double* gpuBuffPtr,double v, long size){
    kernelSetBuffer<<<size/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(gpuBuffPtr,v,size);
}

__host__ void cudaDownloadBuffer(double* gpuBuffPtr, double * cpuBuffPtr,long size){
    gpuErrchk(cudaMemcpy(cpuBuffPtr,gpuBuffPtr,size*sizeof(double),cudaMemcpyDeviceToHost));
}
__host__ void cudaFreeBuffer(double* gpuBuffPtr){
    gpuErrchk(cudaFree(gpuBuffPtr));
}

#endif
