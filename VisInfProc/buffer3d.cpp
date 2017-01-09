#include "buffer3d.h"

#include <QString>
#include <QImage>
#include <QColor>
#include "helper.h"
#include <assert.h>

Buffer3D::Buffer3D():BaseBuffer()
{
    sx = sy = sz = 0;
}

Buffer3D::Buffer3D(int sx, int sy, int sz):BaseBuffer()
{
    assert(sx > 0);
    assert(sy > 0);
    assert(sz > 0);
    this->sx = sx;
    this->sy = sy;
    this->sz = sz;

    itemCnt = sx*sy*sz;
    createCPUBuffer(itemCnt);
    memset(cpuBuffer,0,itemCnt*sizeof(float));
    cpuValid = true;
}
Buffer3D::Buffer3D(const Buffer3D& other):BaseBuffer()
{
    copyFrom(other);
    sx = other.getSizeX();
    sy = other.getSizeY();
    sz = other.getSizeZ();
}

Buffer3D &Buffer3D::operator=(const Buffer3D& other)
{
    BaseBuffer::operator=(other);
    sx = other.getSizeX();
    sy = other.getSizeY();
    sz = other.getSizeZ();
    return *this;
}

float& Buffer3D::operator()(int x, int y, int z)
{
    assert(x >= 0 && x < sx);
    assert(y >= 0 && y < sy);
    assert(z >= 0 && z < sz);

    if(!cpuValid)
        downloadBuffer();

    gpuValid = false;

    return cpuBuffer[z*sx*sy + y*sx + x];
}

void Buffer3D::resize(int sx, int sy, int sz)
{
    if(sx == this->sx
            && sy == this->sy
             && sz == this->sz)
        return;

    this->sx = sx;
    this->sy = sy;
    this->sz = sz;
    itemCnt = sx*sy*sz;
    createCPUBuffer(itemCnt);
    createGPUBuffer(itemCnt);
    cpuValid = true;
    gpuValid = true;
}
QImage Buffer3D::toImageXY(int pos, float min, float max) const
{
    assert(pos >= 0 && pos < sz);
    qWarning("Don't use ! Port to GPU");

    if(!cpuValid)
        downloadBuffer();

    QImage img(sx,sy,QImage::Format_RGB888);

    float mx = max;
    float mn = min;
    if(min == 0 && max == 0){
        mx = *std::max_element(cpuBuffer,cpuBuffer+sz*sx*sy);
        mn = *std::min_element(cpuBuffer,cpuBuffer+sz*sx*sy);
    }

    float* buffPtr = cpuBuffer+pos*sx*sy;

#pragma omp parallel for
    for(int y = 0; y < sy; y++){
        uchar* ptr = img.scanLine(y);
        float* buffPtrY = buffPtr + y*sx;
        for(int x = 0; x < sx*3; x+=3){

            Helper::pseudoColor(buffPtrY[x/3],mn,mx,
                    &ptr[x+0],&ptr[x+1],&ptr[x+2]);
        }
    }

    return img;
}
QImage Buffer3D::toImageXZ(int pos, float min, float max ) const
{
    assert(pos >= 0 && pos < sy);
    qWarning("Don't use ! Port to GPU");
    if(!cpuValid)
        downloadBuffer();

    QImage img(sx,sz,QImage::Format_RGB888);
    float mx = max;
    float mn = min;
    if(min == 0 && max == 0){
        mx = *std::max_element(cpuBuffer,cpuBuffer+sz*sx*sy);
        mn = *std::min_element(cpuBuffer,cpuBuffer+sz*sx*sy);
    }

    int sxy = sx*sy;
#pragma omp parallel for
    for(int z = 0; z < sz; z++){
        uchar* ptr = img.scanLine(z);
        float* buffPtr = cpuBuffer + z*sxy + pos*sx;
        for(int x = 0; x < sx*3; x+=3){
            Helper::pseudoColor(buffPtr[x/3],mn,mx,
                    &ptr[x+0],&ptr[x+1],&ptr[x+2]);
        }
    }

    return img;
}
QImage Buffer3D::toImageYZ(int pos, float min, float max) const
{
    assert(pos >= 0 && pos < sx);
    qWarning("Don't use ! Port to GPU");

    if(!cpuValid)
        downloadBuffer();

    QImage img(sy,sz,QImage::Format_RGB888);
    float mx = max;
    float mn = min;
    if(min == 0 && max == 0){
        mx = *std::max_element(cpuBuffer,cpuBuffer+sz*sx*sy);
        mn = *std::min_element(cpuBuffer,cpuBuffer+sz*sx*sy);
    }
    int sxy = sx*sy;

#pragma omp parallel for
    for(int z = 0; z < sz; z++){
        uchar* ptr = img.scanLine(z);
        float* buffPtr = cpuBuffer + z*sxy + pos;
        for(int y = 0; y< sy*3; y+=3){

            Helper::pseudoColor(buffPtr[y/3*sx],mn,mx,
                    &ptr[y+0],&ptr[y+1],&ptr[y+2]);
        }
    }
    return img;
}
