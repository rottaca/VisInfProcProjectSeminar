#ifndef BASEBUFFER_H
#define BASEBUFFER_H

#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_helper.h"
#include <nvToolsExt.h>

class BaseBuffer
{
public:
    BaseBuffer();
    virtual ~BaseBuffer();

    virtual BaseBuffer& operator=(const BaseBuffer &other);

    virtual BaseBuffer& operator-=(const BaseBuffer &other);
    virtual BaseBuffer& operator+=(const BaseBuffer &other);

    size_t getBufferItemCnt() const
    {
        return itemCnt;
    }

    void setCudaStream(cudaStream_t stream)
    {
        cudaStream = stream;
    }

    void copyFrom(const BaseBuffer& other);
    void downloadBuffer() const;
    void uploadBuffer() const;

    bool isGPUValid() const
    {
        return gpuValid;
    }
    bool isCPUValid() const
    {
        return cpuValid;
    }

    float* getCPUPtr() const
    {
        if(!cpuValid && gpuValid)
            downloadBuffer();
        gpuValid = false;
        return cpuBuffer;
    }

    float* getGPUPtr() const
    {
        if(!gpuValid && cpuValid)
            uploadBuffer();
        cpuValid = false;
        return gpuBuffer;
    }

    float getMax() const
    {
        if(!cpuValid) {
            if(gpuValid)
                downloadBuffer();
            else
                return 0;
        }

        return *std::max_element(cpuBuffer, cpuBuffer + getBufferItemCnt());
    }
    float getMin() const
    {
        if(!cpuValid) {
            if(gpuValid)
                downloadBuffer();
            else
                return 0;
        }
        return *std::min_element(cpuBuffer, cpuBuffer + getBufferItemCnt());
    }

    void fill(float v)
    {

        if(gpuBuffer == NULL) {
            createGPUBuffer(itemCnt);
        }
        cudaSetDoubleBuffer(gpuBuffer,v,itemCnt);
        cpuValid = false;
        gpuValid = true;
    }

protected:
    void createCPUBuffer(size_t sz);
    void createGPUBuffer(size_t sz);

protected:
    // Mutable: Allows change on const object
    // Necessary to download/upload data, this does not change the object
    // in a logical way
    mutable float *cpuBuffer;
    mutable float *gpuBuffer;
    mutable unsigned char   *gpuImage;

    mutable bool gpuValid,cpuValid;
    size_t itemCnt;

    cudaStream_t cudaStream;
};

#endif // BASEBUFFER_H
