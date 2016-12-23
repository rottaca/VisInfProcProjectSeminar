#ifndef CUDA_CONVOLUTION3D_H
#define CUDA_CONVOLUTION3D_H

extern void cudaConvolution3D(double* gpuBuffer,
                              int writeIdx, int bsx, int bsy, int bsz,
                              double* gpuFilter, int fsx, int fsy, int fsz,
                              int px, int py);

#endif // CUDA_CONVOLUTION3D_H

