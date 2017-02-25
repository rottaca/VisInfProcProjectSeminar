
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_settings.h"
#include "cuda_helper.h"

__global__ void kernelComputeFlow(int n,
                                  float* gpuEnergy, float* gpuDir, float* gpuSpeed,
                                  float** gpuArrGpuEnergies,
                                  float* gpuArrOrientations, int orientationCnt,
                                  float* gpuArrSpeeds, int speedCnt,
                                  float minEnergy)
{
    int pixelIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if(pixelIdx < n) {
        float resDir = 0;
        float resSpeed = 0;
        float resEnergy = 0;

#ifdef DISABLE_INTERPOLATION
        float resEnergyX = 0;
        float resEnergyY = 0;
        int energyIdx = 0;
        for(int i = 0; i < speedCnt; i++) {
            float localEnergyX = 0;
            float localEnergyY = 0;
            float localEnergy = 0;
            for(int j = 0; j  < orientationCnt; j++) {
                localEnergyX += gpuArrGpuEnergies[energyIdx][pixelIdx]*cos(gpuArrOrientations[j]);
                localEnergyY += gpuArrGpuEnergies[energyIdx][pixelIdx]*sin(gpuArrOrientations[j]);
                energyIdx++;
            }
            localEnergy = sqrt(localEnergyX*localEnergyX+localEnergyY*localEnergyY);
            if(localEnergy > resEnergy) {
                resEnergy = localEnergy;
                resEnergyX = localEnergyX;
                resEnergyY = localEnergyY;
                resSpeed = gpuArrSpeeds[i];
            }
        }
        resDir = atan2(resEnergyY,resEnergyX);
#else
        float exp = 2;
        float energySum = 0;
        float speedX = 0;
        float speedY = 0;

        // Calcualte energy sum
        for(int i = 0; i < speedCnt*orientationCnt; i++) {
            energySum += gpuArrGpuEnergies[i][pixelIdx];
        }

        float scaleSum = 0;
        for(int i = 0; i < speedCnt*orientationCnt; i++) {
            scaleSum += pow(gpuArrGpuEnergies[i][pixelIdx]/energySum,exp);
        }

        for(int j = 0; j  < orientationCnt; j++) {
            float orientation = gpuArrOrientations[j];
            for(int i = 0; i < speedCnt; i++) {
                float speed = gpuArrSpeeds[i];
                int idx = i*orientationCnt+j;
                float energy = gpuArrGpuEnergies[idx][pixelIdx];
                // calculate scaling factor
                float scale = pow(gpuArrGpuEnergies[idx][pixelIdx]/energySum,exp)/scaleSum;
                //float scale = gpuArrGpuEnergies[idx][pixelIdx]/energySum;

                resEnergy += energy*scale;
                speedX += speed*cos(orientation)*scale;
                speedY += speed*sin(orientation)*scale;
            }
        }
        resDir = atan2f(speedY,speedX);
        resSpeed = sqrtf(speedX*speedX+speedY*speedY);
#endif

        if(resEnergy >= minEnergy) {
            // Keep orientation between 0 and 2*PI
            if(resDir < 0) resDir += 2*M_PI;
            gpuDir[pixelIdx] = resDir;
            gpuEnergy[pixelIdx] = resEnergy;
            gpuSpeed[pixelIdx] = resSpeed;

        } else {
            gpuDir[pixelIdx] = 0;
            gpuEnergy[pixelIdx] = 0;
            gpuSpeed[pixelIdx] = 0;
        }
    }
}
/**
 * @brief cudaComputeFlowEnergyAndDir Takes convolution buffers from all orientations
 *                                    and computes the flow energy and direction
 * @param sx
 * @param sy
 * @param gpuEnergy
 * @param gpuDir
 * @param gpuArrGpuEnergy Array of pointer to buffers
 * @param gpuArrOrientations
 * @param orientationCnt
 * @param speed
 * @param stream
 */
__host__ void cudaComputeFlow(int sx, int sy,
                              float* gpuEnergy, float* gpuDir, float* gpuSpeed,
                              float** gpuArrGpuEnergies,
                              float* gpuArrOrientations, int orientationCnt,
                              float* gpuArrSpeeds, int speedCnt,
                              float minEnergy,
                              cudaStream_t stream)
{
    int n = sx*sy;
    size_t blocks = ceilf((float)n/THREADS_PER_BLOCK);
    kernelComputeFlow<<<blocks,THREADS_PER_BLOCK,0,stream>>>(
        n,
        gpuEnergy,gpuDir,gpuSpeed,
        gpuArrGpuEnergies,
        gpuArrOrientations,orientationCnt,
        gpuArrSpeeds,speedCnt,
        minEnergy);
}

__global__ void kernelFlowToRGB(float* gpuEnergy, float* gpuDir, char *gpuImage,
                                int n,
                                float maxLength)
{

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n) {
        float length = gpuEnergy[idx];
        float h = RAD2DEG(gpuDir[idx]);

        // Mod 360
        h = fmod(h,360.0f);
        if(h < 0) h += 360;

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
        switch(i) {
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
            r = v;
            g = p;
            b = q;
            break;
        // This should not happen ! Set color to black
        default:
            r = 0;
            g = 0;
            b = 0;
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
