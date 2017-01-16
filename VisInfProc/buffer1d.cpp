#include "buffer1d.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <cstring>

Buffer1D::Buffer1D():BaseBuffer()
{
}

Buffer1D::Buffer1D(int size):BaseBuffer()
{
    assert(size > 0);
    itemCnt = size;
    createCPUBuffer(itemCnt);
    memset(cpuBuffer,0,itemCnt*sizeof(float));
    cpuValid = true;
}

Buffer1D::Buffer1D(const Buffer1D& other):BaseBuffer()
{
    copyFrom(other);
}

Buffer1D& Buffer1D::operator=(const Buffer1D &other)
{
    BaseBuffer::operator=(other);
    itemCnt = other.getSize();
    return *this;
}

float& Buffer1D::operator()(int i)
{
    assert(i >= 0 && i < itemCnt);
    if(!cpuValid)
        downloadBuffer();
    gpuValid = false;
    return cpuBuffer[i];
}
