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
extern void* cudaCreateBuffer(long size);
extern void cudaUploadBuffer(void* cpuBuffPtr, void * gpuBuffPtr,long size);
extern void cudaDownloadBuffer(void* gpuBuffPtr, void * cpuBuffPtr,long size);
extern void cudaFreeBuffer(void* gpuBuffPtr);
extern void cudaCopyBuffer(void* gpuBuffPtrDest, void * gpuBuffPtrSrc,long size);
extern void cudaSetDoubleBuffer(double* gpuBuffPtr,double v, long size);

#endif // CUDA_HELPER_H

