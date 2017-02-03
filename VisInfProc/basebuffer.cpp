#include "basebuffer.h"

#include <cuda_runtime.h>
#include <cuda_helper.h>
#include <assert.h>
#include <cstring>
#include <QDebug>

BaseBuffer::BaseBuffer()
{
    cudaStream = 0;
    cpuBuffer = NULL;
    gpuBuffer = NULL;
    gpuImage = NULL;
    cpuValid = false;
    gpuValid = false;
    itemCnt = 0;
}

BaseBuffer::~BaseBuffer()
{
    if(gpuBuffer != NULL) {
        cudaFree(gpuBuffer);
        gpuBuffer = NULL;
    }
    if(gpuImage != NULL) {
        cudaFree(gpuImage);
        gpuImage = NULL;
    }
    if(cpuBuffer != NULL) {
        delete[] cpuBuffer;
        cpuBuffer = NULL;
    }
}

void BaseBuffer::copyFrom(const BaseBuffer& other)
{
    if(this == &other)
        return;

    size_t szNew = other.getBufferItemCnt();
    // Does the size differ ?
    if(szNew != itemCnt) {
        // delete my buffers on cpu and gpu
        // And reallocate both with the same size
        createCPUBuffer(szNew);
        createGPUBuffer(szNew);
        itemCnt = szNew;
    }
    // Copy gpu or cpu, depending on which one is available
    // Prefer gpu buffer
    if(other.isGPUValid()) {
        if(gpuBuffer == NULL)
            createGPUBuffer(szNew);

        cudaCopyBuffer(gpuBuffer,other.getGPUPtr(),szNew*sizeof(float),cudaStream);
        gpuValid = true;
        cpuValid = false;
    } else if(other.isCPUValid()) {

        if(cpuBuffer == NULL)
            createCPUBuffer(szNew);

        memcpy(cpuBuffer,other.getCPUPtr(),szNew*sizeof(float));
        gpuValid = false;
        cpuValid = true;
    } else {
        qDebug("Buffer empty, can't copy!");
    }
}

BaseBuffer& BaseBuffer::operator=(const BaseBuffer &other)
{
    copyFrom(other);

    return *this;
}

BaseBuffer& BaseBuffer::operator-=(const BaseBuffer &other)
{
    assert(other.getBufferItemCnt() == getBufferItemCnt());
    if(!cpuValid)
        downloadBuffer();

    gpuValid = false;
    float * ptrOther = other.getCPUPtr();
    for(size_t i = 0; i < getBufferItemCnt(); i++) {
        cpuBuffer[i] -= ptrOther[i];
    }
    return *this;
}

BaseBuffer& BaseBuffer::operator+=(const BaseBuffer &other)
{
    assert(other.getBufferItemCnt() == getBufferItemCnt());
    if(!cpuValid)
        downloadBuffer();

    gpuValid = false;
    float * ptrOther = other.getCPUPtr();
    for(size_t i = 0; i < getBufferItemCnt(); i++) {
        cpuBuffer[i] += ptrOther[i];
    }
    return *this;
}

void BaseBuffer::downloadBuffer() const
{
    assert(gpuBuffer != NULL);
    assert(gpuValid);

    if(cpuBuffer == NULL)
        cpuBuffer = new float[itemCnt];

    cudaDownloadBuffer(gpuBuffer,cpuBuffer,itemCnt*sizeof(float),cudaStream);
    cudaStreamSynchronize(cudaStream);
    cpuValid = true;
    //qDebug("Downloading buffer");
}
void BaseBuffer::uploadBuffer() const
{
    assert(cpuBuffer != NULL);
    assert(cpuValid);

    if(gpuBuffer == NULL)
        gpuBuffer = static_cast<float*>(cudaCreateBuffer(itemCnt*sizeof(float)));

    cudaUploadBuffer(cpuBuffer,gpuBuffer,itemCnt*sizeof(float),cudaStream);
    cudaStreamSynchronize(cudaStream);
    gpuValid = true;
    //qDebug("Uploading buffer");
}

void BaseBuffer::createCPUBuffer(size_t sz)
{
    if(cpuBuffer != NULL)
        delete[] cpuBuffer;
    cpuBuffer = new float[sz];
}

void BaseBuffer::createGPUBuffer(size_t sz)
{
    if(gpuBuffer != NULL)
        cudaFreeBuffer(gpuBuffer);
    gpuBuffer = static_cast<float*>(cudaCreateBuffer(sz*sizeof(float)));
}
