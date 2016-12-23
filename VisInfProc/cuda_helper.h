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

extern double* cudaCreateDoubleBuffer(long size);
extern void cudaUploadDoubleBuffer(double* cpuBuffPtr, double * gpuBuffPtr,long size);
extern void cudaSetDoubleBuffer(double* gpuBuffPtr,double v, long size);
extern void cudaDownloadDoubleBuffer(double* gpuBuffPtr, double * cpuBuffPtr, long size);
extern void cudaFreeDoubleBuffer(double* gpuBuffPtr);

#endif // CUDA_HELPER_H

