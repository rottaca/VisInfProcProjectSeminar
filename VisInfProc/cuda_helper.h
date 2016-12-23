#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define gpuKernelErrorCheck() { cudaDeviceSynchronize();\
     cudaError_t error = cudaGetLastError(); \
     if(error != cudaSuccess) \
     {\
       printf("CUDA error: %s\n", cudaGetErrorString(error));\
       exit(-1);\
     }\
}

extern double* cudaCreateBuffer(long size);
extern void cudaUploadBuffer(double* cpuBuffPtr, double * gpuBuffPtr,long size);
extern void cudaSetBuffer(double* gpuBuffPtr,double v, long size);
extern void cudaDownloadBuffer(double* gpuBuffPtr, double * cpuBuffPtr, long size);
extern void cudaFreeBuffer(double* gpuBuffPtr);

#endif // CUDA_HELPER_H

