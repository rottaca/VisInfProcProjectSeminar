#ifndef BASEBUFFER_H
#define BASEBUFFER_H

#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_helper.h"

class BaseBuffer
{
public:
    BaseBuffer();
    virtual ~BaseBuffer();

    virtual BaseBuffer& operator=(const BaseBuffer &other);

    virtual BaseBuffer& operator-=(const BaseBuffer &other);
    virtual BaseBuffer& operator+=(const BaseBuffer &other);

    long getBufferItemCnt() const{
        return itemCnt;
    }

    void copyFrom(const BaseBuffer& other);
    void downloadBuffer() const;
    void uploadBuffer() const;

    bool isGPUValid() const{
        return gpuValid;
    }
    bool isCPUValid() const{
        return cpuValid;
    }

    double* getCPUPtr() const{
        if(!cpuValid && gpuValid)
            downloadBuffer();
        gpuValid = false;
        return cpuBuffer;
    }

    double* getGPUPtr() const{
        if(!gpuValid && cpuValid)
            uploadBuffer();
        cpuValid = false;
        return gpuBuffer;
    }

    double getMax() const{
        if(!cpuValid){
            if(gpuValid)
                downloadBuffer();
            else
                return 0;
        }

        return *std::max_element(cpuBuffer, cpuBuffer + getBufferItemCnt());
    }
    double getMin() const{
        if(!cpuValid){
            if(gpuValid)
                downloadBuffer();
            else
                return 0;
        }
        return *std::min_element(cpuBuffer, cpuBuffer + getBufferItemCnt());
    }

    void fill(double v){

        if(gpuBuffer == NULL){
            createGPUBuffer(itemCnt);
        }
        cudaSetDoubleBuffer(gpuBuffer,v,itemCnt);
        cpuValid = false;
        gpuValid = true;
    }

protected:
    void createCPUBuffer(long sz);
    void createGPUBuffer(long sz);

protected:
    // Mutable: Allows change on const object
    // Necessary to download/upload data, this does not change the object
    // in a logical way
    mutable double *cpuBuffer;
    mutable double *gpuBuffer;
    mutable unsigned char   *gpuImage;

    mutable bool gpuValid,cpuValid;
    mutable long itemCnt;
};

#endif // BASEBUFFER_H
