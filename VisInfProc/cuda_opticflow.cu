
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_settings.h"
#include "cuda_helper.h"


__global__ void kernelComputeOpticFlow(int n,
                                       float* gpuFlowX,float* gpuFlowY,
                                       float** gpuEnergy,
                                       float* orientations, int orientationCnt,
                                       float *speeds, int speedCnt){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n){
//        // TODO only store the 3 important
//        //float energies[speedCnt][orientationCnt];
//        int maxSpeedIdx = 0;
//        float maxSpeedEnergy = 0;
//        // Read all energies from all buffers
//        // Read combined energy for each speed
//        // Find maximum energy
//        for(int i = 0; i  < speedCnt; i++){
//            float speedEnergy = 0;
//            for(int j = 0; j  < orientationCnt; j++){
//                //energies[i][j] = gpuEnergy[i*orientationCnt + j][idx];
//                speedEnergy += gpuEnergy[i*orientationCnt + j][idx];//energies[i][j];
//            }
//            // Find maximum energy
//            if(maxSpeedEnergy < speedEnergy){
//                maxSpeedEnergy = speedEnergy;
//                maxSpeedIdx = i;
//            }
//        }

//        // Polynomal interpolation
//        // Fit polynom y = ax^2 + bx + c
//        // Source: http://math.stackexchange.com/questions/680646/get-polynomial-function-from-3-points
//        // and: http://www.wikihow.com/Find-the-Maximum-or-Minimum-Value-of-a-Quadratic-Function-Easily
//        if(speedCnt >= 3){
//            // Find the three closest energies
//            int i1,i2,i3;
//            // Maximum is at the beginning ?
//            if(maxSpeedIdx == 0){
//                i1 = 0;
//                i2 = i1+1;
//                i3 = i2+1;
//            }
//            // Maximum is at the end ?
//            else if(maxSpeedIdx == speedCnt -1){
//                i1 = speedCnt-1;
//                i2 = i1-1;
//                i3 = i2-1;
//            }
//            // Maximum is inbetween
//            else{
//                i1 = maxSpeedIdx - 1;
//                i2 = maxSpeedIdx;
//                i3 = maxSpeedIdx + 1;
//            }


//            // Compute polynom coefficients a,b and c
//            // Xi's are the speeds
//            // Yi's are the energies
//            float a = speeds[i1]*(speedEnergy[i3] - speedEnergy[i2]) +
//                speeds[i2]*(speedEnergy[i1] - speedEnergy[i3]) +
//                speeds[i3]*(speedEnergy[i2] - speedEnergy[i1]) /
//                ((speeds[i1]-speeds[i2])*
//                 (speeds[i1]-speeds[i3])*
//                 (speeds[i2]-speeds[i3]));
//            float b = (speedEnergy[i2] - speedEnergy[i1])/(speeds[i2]-speeds[i1]) - a*(speeds[i1]+speeds[i2]);
//            float c = speedEnergy[i1] - a*speeds[i1]*speeds[i1] - b*speeds[i1];


//            // Check for max or min
//            if(a > 0){
//                // Find maximum energy by finding maximum of polynom y = ax^2+bx+c
//                // Max speed is the length of our flow vector
//                float maxSpeed = -b/(2*a);
//                float maxEnergy = a*maxSpeed*maxSpeed + b*maxSpeed + c;

//                // just combine all orientations from the buffer with the highest energy
//                for(int i = 0; i  < orientationCnt; i++)
//                {
//                    localFlowX += speedEnergy[maxSpeedIdx]*cos(orientations[i]);
//                    localFlowY += speedEnergy[maxSpeedIdx]*sin(orientations[i]);
//                }
//            }
//            // we don't have a maximum
//            else{
//                // just combine all orientations from the buffer with the highest energy
//                for(int i = 0; i  < orientationCnt; i++)
//                {
//                    localFlowX += speedEnergy[maxSpeedIdx]*cos(orientations[i]);
//                    localFlowY += speedEnergy[maxSpeedIdx]*sin(orientations[i]);
//                }
//            }
 //       }
        // Take maximum speed, no interpolation
//        else{

        float localFlowX = 0;
        float localFlowY = 0;
        float energy = 0;
        int maxIdx = 0;
        for(int i = 0; i < speedCnt; i++)
        {
            float fX = 0;
            float fY = 0;

            for(int j = 0; j  < orientationCnt; j++)
            {
                fX += gpuEnergy[i*orientationCnt + j][idx]*cos(orientations[j]);
                fY += gpuEnergy[i*orientationCnt + j][idx]*sin(orientations[j]);
            }
            // Vector norm
            float e = sqrt(fX*fX+fY*fY);
            if(e > energy){
                energy = e;
                localFlowX = fX;
                localFlowY = fY;
                maxIdx = i;
            }
        }
//        }
        if(energy > 0.1)
        {
            float speed = speeds[maxIdx];
            //gpuFlowX[idx] = localFlowX/energy*speed;
            //gpuFlowY[idx] = localFlowY/energy*speed;
            gpuFlowX[idx] = localFlowX;
            gpuFlowY[idx] = localFlowY;
        }else{
            gpuFlowX[idx] = 0;
            gpuFlowY[idx] = 0;
        }
    }
}

__host__ void cudaComputeOpticFlow(int sx, int sy,
                                   float* gpuFlowX, float* gpuFlowY,
                                   float** gpuArrGpuEnergy,
                                   float* gpuArrOrientations, int orientationCnt,
                                   float *speeds, int speedCnt,
                                   cudaStream_t stream)
{
    int n = sx*sy;
    long blocks = ceil((float)n/THREADS_PER_BLOCK);
    kernelComputeOpticFlow<<<blocks,THREADS_PER_BLOCK,0,stream>>>(
                         n,
                         gpuFlowX,gpuFlowY,
                         gpuArrGpuEnergy,
                         gpuArrOrientations,orientationCnt,
                         speeds,speedCnt);
}

__global__ void kernelFlowToRGB(float* gpuFlowX, float* gpuFlowY, char *gpuImage,
                                int n,
                                float maxLength){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n){
        float fx = gpuFlowX[idx];
        float fy = gpuFlowY[idx];
        float length = sqrt(fx*fx+fy*fy);
        float h = atan2(fy,fx)/(2*M_PI)*360;
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
        switch(i){
            case 0: r = v; g = t; b = p; break;
            case 1: r = q; g = v; b = p; break;
            case 2: r = p; g = v; b = t; break;
            case 3: r = p; g = q; b = v; break;
            case 4: r = t; g = p; b = v; break;
            case 5:
            default:r = v; g = p; b = q; break;
        }

        gpuImage[3*idx    ] = r*255;
        gpuImage[3*idx + 1] = g*255;
        gpuImage[3*idx + 2] = b*255;
    }
}

__host__ void cudaFlowToRGB(float* gpuFlowX, float* gpuFlowY, char *gpuImage,
                            int sx, int sy,
                            float maxLength, cudaStream_t stream){
    int n = sx*sy;
    long blocks = ceil((float)n/THREADS_PER_BLOCK);
    kernelFlowToRGB<<<blocks,THREADS_PER_BLOCK,0,stream>>>(
                         gpuFlowX,gpuFlowY,gpuImage,
                         n,
                         maxLength);
}
