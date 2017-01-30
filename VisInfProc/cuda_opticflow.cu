
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_settings.h"
#include "cuda_helper.h"


__global__ void kernelComputeFlowEnergyAndDir(int n,
        float* gpuEnergy,float* gpuDir,
        float** gpuArrGpuEnergy,
        float* orientations, int orientationCnt,
        float speed)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n)
        {
            float localFlowX = 0;
            float localFlowY = 0;
            for(int j = 0; j  < orientationCnt; j++)
                {
                    localFlowX += gpuArrGpuEnergy[j][idx]*cos(orientations[j]);
                    localFlowY += gpuArrGpuEnergy[j][idx]*sin(orientations[j]);
                }
            gpuDir[idx] = atan2(localFlowY,localFlowX);
            gpuEnergy[idx] = sqrt(localFlowX*localFlowX+localFlowY*localFlowY);
        }
}

__host__ void cudaComputeFlowEnergyAndDir(int sx, int sy,
        float* gpuEnergy, float* gpuDir,
        float** gpuArrGpuEnergy,
        float* gpuArrOrientations, int orientationCnt,
        float speed,
        cudaStream_t stream)
{
    int n = sx*sy;
    size_t blocks = ceil((float)n/THREADS_PER_BLOCK);
    kernelComputeFlowEnergyAndDir<<<blocks,THREADS_PER_BLOCK,0,stream>>>(
        n,
        gpuEnergy,gpuDir,
        gpuArrGpuEnergy,
        gpuArrOrientations,orientationCnt,
        speed);
}

__global__ void kernelFlowToRGB(float* gpuEnergy, float* gpuDir, char *gpuImage,
                                int n,
                                float maxLength)
{

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n)
        {
            float length = gpuEnergy[idx];
            float h = gpuDir[idx]/(2*M_PI)*360;
            if(h < 0)
                h+= 360;
            else if(h>= 360)
                h = 0;
            float s = length/maxLength;
            if(s > 1)
                s = 1;

            float v = 1;

            h /= 60.0f;
            int i = h;
            float ff = h-i;
            float p = v*(1.0f-s);
            float q = v*(1.0f - (s*ff));
            float t = v*(1.0f - (s* (1.0f-ff)));

            float r,g,b;
            switch(i)
                {
                case 0:
                    r = v;
                    g = t;
                    b = p;
                    break;
                case 1:
                    r = q;
                    g = v;
                    b = p;
                    break;
                case 2:
                    r = p;
                    g = v;
                    b = t;
                    break;
                case 3:
                    r = p;
                    g = q;
                    b = v;
                    break;
                case 4:
                    r = t;
                    g = p;
                    b = v;
                    break;
                case 5:
                default:
                    r = v;
                    g = p;
                    b = q;
                    break;
                }

            gpuImage[3*idx    ] = r*255;
            gpuImage[3*idx + 1] = g*255;
            gpuImage[3*idx + 2] = b*255;
        }
}

__host__ void cudaFlowToRGB(float* gpuEnergy, float* gpuDir, char *gpuImage,
                            int sx, int sy,
                            float maxLength, cudaStream_t stream)
{
    int n = sx*sy;
    size_t blocks = ceil((float)n/THREADS_PER_BLOCK);
    kernelFlowToRGB<<<blocks,THREADS_PER_BLOCK,0,stream>>>(
        gpuEnergy,gpuDir,gpuImage,
        n,
        maxLength);
}
