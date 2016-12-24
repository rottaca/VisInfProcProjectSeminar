#include "buffer3d.h"

#include <QString>
#include <QImage>
#include <QColor>
#include "helper.h"
#include <assert.h>

Buffer3D::Buffer3D()
{
    sx = sy = sz = 0;
    buffer = NULL;
}

Buffer3D::Buffer3D(int sx, int sy, int sz)
{
    this->sx = sx;
    this->sy = sy;
    this->sz = sz;

    size_t s = sx*sy*sz;
    buffer = new double[s];
    memset(buffer,0,s*sizeof(double));
}
Buffer3D::Buffer3D(const Buffer3D& other)
{
    sx = other.getSizeX();
    sy = other.getSizeY();
    sz = other.getSizeZ();

    size_t s = sx*sy*sz;
    buffer = new double[s];
    memcpy(buffer,other.getBuff(),s*sizeof(double));
}

Buffer3D &Buffer3D::operator=(const Buffer3D& other)
{
    if(this != &other){
        resize(other.getSizeX(),other.getSizeY(),other.getSizeZ());
        memcpy(buffer,other.getBuff(),sx*sy*sz*sizeof(double));
    }

    return *this;
}

double& Buffer3D::operator()(int x, int y, int z)
{
    assert(x >= 0 && x < sx);
    assert(y >= 0 && y < sy);
    assert(z >= 0 && z < sz);
    return buffer[z*sx*sy + y*sx + x];
}

double Buffer3D::operator()(int x, int y, int z) const
{
    assert(x >= 0 && x < sx);
    assert(y >= 0 && y < sy);
    assert(z >= 0 && z < sz);
    return buffer[z*sx*sy + y*sx + x];
}

Buffer3D Buffer3D::operator-(Buffer3D& b) const
{
    assert(sx == b.getSizeX() && sy == b.getSizeY() && sz == b.getSizeZ());
    Buffer3D buff(sx,sy,sz);
    double *ptrRes = buff.getBuff();
    double *ptrB = b.getBuff();
    for(int i = 0; i < sx*sy*sz; i++){
        ptrRes[i] = buffer[i]-ptrB[i];
    }

    return buff;
}

Buffer3D Buffer3D::operator+(Buffer3D& b) const
{
    assert(sx == b.getSizeX() && sy == b.getSizeY() && sz == b.getSizeZ());
    Buffer3D buff(sx,sy,sz);
    double *ptrRes = buff.getBuff();
    double *ptrB = b.getBuff();
    for(int i = 0; i < sx*sy*sz; i++){
        ptrRes[i] = buffer[i]+ptrB[i];
    }
    return buff;
}

Buffer3D::~Buffer3D()
{
    if(buffer != NULL){
        delete[] buffer;
        buffer = NULL;
    }
}
void Buffer3D::resize(int sx, int sy, int sz)
{
    if(sx == this->sx && sy == this->sy && sz == this->sz)
        return;

    if(buffer != NULL)
        delete[] buffer;

    this->sx = sx;
    this->sy = sy;
    this->sz = sz;
    size_t s = sx*sy*sz;
    buffer = new double[s];
}
QImage Buffer3D::toImageXY(int pos, double min, double max) const
{
    assert(pos >= 0 && pos < sz);
    QImage img(sx,sy,QImage::Format_RGB888);

    double mx = max;
    double mn = min;
    if(min == 0 && max == 0){
        mx = *std::max_element(buffer,buffer+sz*sx*sy);
        mn = *std::min_element(buffer,buffer+sz*sx*sy);
    }

    double* buffPtr = buffer+pos*sx*sy;

#pragma omp parallel for
    for(int y = 0; y < sy; y++){
        uchar* ptr = img.scanLine(y);
        double* buffPtrY = buffPtr + y*sx;
        for(int x = 0; x < sx*3; x+=3){

            Helper::pseudoColor(buffPtrY[x/3],mn,mx,
                    &ptr[x+0],&ptr[x+1],&ptr[x+2]);
        }
    }

    return img;
}
QImage Buffer3D::toImageXZ(int pos, double min, double max ) const
{
    assert(pos >= 0 && pos < sy);
    QImage img(sx,sz,QImage::Format_RGB888);
    double mx = max;
    double mn = min;
    if(min == 0 && max == 0){
        mx = *std::max_element(buffer,buffer+sz*sx*sy);
        mn = *std::min_element(buffer,buffer+sz*sx*sy);
    }

    int sxy = sx*sy;
#pragma omp parallel for
    for(int z = 0; z < sz; z++){
        uchar* ptr = img.scanLine(z);
        double* buffPtr = buffer + z*sxy + pos*sx;
        for(int x = 0; x < sx*3; x+=3){
            Helper::pseudoColor(buffPtr[x/3],mn,mx,
                    &ptr[x+0],&ptr[x+1],&ptr[x+2]);
        }
    }

    return img;
}
QImage Buffer3D::toImageYZ(int pos, double min, double max) const
{
    assert(pos >= 0 && pos < sx);
    QImage img(sy,sz,QImage::Format_RGB888);
    double mx = max;
    double mn = min;
    if(min == 0 && max == 0){
        mx = *std::max_element(buffer,buffer+sz*sx*sy);
        mn = *std::min_element(buffer,buffer+sz*sx*sy);
    }
    int sxy = sx*sy;

#pragma omp parallel for
    for(int z = 0; z < sz; z++){
        uchar* ptr = img.scanLine(z);
        double* buffPtr = buffer + z*sxy + pos;
        for(int y = 0; y< sy*3; y+=3){

            Helper::pseudoColor(buffPtr[y/3*sx],mn,mx,
                    &ptr[y+0],&ptr[y+1],&ptr[y+2]);
        }
    }
    return img;
}
