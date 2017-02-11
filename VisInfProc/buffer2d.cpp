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
    memset(cpuBuffer,0,itemCnt*sizeof(float));
    cpuValid = true;
    qImage = QImage(sx,sy,QImage::Format_RGB888);
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

float& Buffer2D::operator()(int x, int y)
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
    qImage = QImage(sx,sy,QImage::Format_RGB888);
}

QImage &Buffer2D::toImage(float min, float max) const
{
    float mx = max;
    float mn = min;

#ifdef DEBUG_INSERT_PROFILER_MARKS
    nvtxRangeId_t id = nvtxRangeStart("2D-Buffer to Image");
#endif
    // Compute max and min on cpu if necessary -> BAD
    if(min == 0 && max == 0) {
        if(!cpuValid)
            downloadBuffer();
        mx = *std::max_element(cpuBuffer,cpuBuffer+sx*sy);
        mn = *std::min_element(cpuBuffer,cpuBuffer+sx*sy);
    }

    // Process data on GPU
    if(!gpuValid)
        uploadBuffer();

    if(gpuImage == NULL)
        gpuImage = static_cast<unsigned char*>(cudaCreateBuffer(itemCnt*3));

    if(qImage.width() != sx || qImage.height() != sy)
        qImage = QImage(sx,sy,QImage::Format_RGB888);

    cuda2DBufferToRGBImage(sx,sy,mn,mx,gpuBuffer,gpuImage,cudaStream);
    cudaDownloadBuffer(gpuImage,qImage.bits(),itemCnt*3,cudaStream);

#ifdef DEBUG_INSERT_PROFILER_MARKS
    nvtxRangeEnd(id);
#endif

    return qImage;
}
