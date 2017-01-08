#include "buffer2d.h"
#include <QString>
#include <QRgb>
#include <QColor>
#include "helper.h"
#include <assert.h>

Buffer2D::Buffer2D():BaseBuffer()
{
    sx = sy = 0;
}

Buffer2D::Buffer2D(int sx, int sy):BaseBuffer()
{
    assert(sx > 0);
    assert(sy > 0);
    this->sx = sx;
    this->sy = sy;
    itemCnt = sx*sy;
    createCPUBuffer(itemCnt);
    memset(cpuBuffer,0,itemCnt*sizeof(double));
    cpuValid = true;
}
Buffer2D::Buffer2D(const Buffer2D& other):BaseBuffer()
{
    copyFrom(other);
    sx = other.getSizeX();
    sy = other.getSizeY();
}

Buffer2D &Buffer2D::operator=(const Buffer2D& other)
{
    BaseBuffer::operator=(other);
    sx = other.getSizeX();
    sy = other.getSizeY();
    return *this;
}

double& Buffer2D::operator()(int x, int y)
{
    assert(x >= 0 && x < sx);
    assert(y >= 0 && y < sy);
    if(!cpuValid)
        downloadBuffer();

    gpuValid = false;

    return cpuBuffer[y*sx + x];
}

void Buffer2D::resize(int sx, int sy)
{
    if(sx == this->sx && sy == this->sy)
        return;

    this->sx = sx;
    this->sy = sy;
    itemCnt = sx*sy;
    createCPUBuffer(itemCnt);
    createGPUBuffer(itemCnt);
    cpuValid = true;
    gpuValid = true;
}

QImage Buffer2D::toImage(double min, double max) const
{
    double mx = max;
    double mn = min;

#ifndef NDEBUG
    nvtxRangeId_t id = nvtxRangeStart("2D-Buffer to Image");
#endif
    // Compute max and min on cpu if necessary -> BAD
    if(min == 0 && max == 0){
        if(!cpuValid)
            downloadBuffer();
        mx = *std::max_element(cpuBuffer,cpuBuffer+sx*sy);
        mn = *std::min_element(cpuBuffer,cpuBuffer+sx*sy);
    }

    // Process data on GPU
    if(!gpuValid)
        uploadBuffer();

    QImage img(sx,sy,QImage::Format_RGB888);

    if(gpuImage == NULL)
        gpuImage = static_cast<unsigned char*>(cudaCreateBuffer(itemCnt*3));

    cuda2DBufferToRGBImage(sx,sy,mn,mx,gpuBuffer,gpuImage,cudaStream);
    cudaDownloadBuffer(gpuImage,img.bits(),itemCnt*3,cudaStream);

#ifndef NDEBUG
    nvtxRangeEnd(id);
#endif

    return img;
}


