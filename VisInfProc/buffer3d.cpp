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
        if(buffer != NULL)
        {
            delete []buffer;
            buffer = NULL;
        }

        sx = other.getSizeX();
        sy = other.getSizeY();
        sz = other.getSizeZ();

        size_t s = sx*sy*sz;
        buffer = new double[s];
        memcpy(buffer,other.getBuff(),s*sizeof(double));
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

    for(int y = 0; y < sy; y++){
        uchar* ptr = img.scanLine(y);
        for(int x = 0; x < sx*3; ){
            QColor rgb = Helper::pseudoColor(buffPtr[y*sx+x/3],mn,mx);
            ptr[x++] = rgb.red();
            ptr[x++] = rgb.green();
            ptr[x++] = rgb.blue();
        }
    }

    uchar* ptr = img.scanLine(0);
    ptr[0] = 0;
    ptr[1] = 0;
    ptr[2] = 0;
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

    for(int z = 0; z < sz; z++){
        uchar* ptr = img.scanLine(z);
        for(int x = 0; x < sx*3; ){
            QColor rgb = Helper::pseudoColor(buffer[z*sx*sy + pos*sx + x/3],mn,mx);
            ptr[x++] = rgb.red();
            ptr[x++] = rgb.green();
            ptr[x++] = rgb.blue();
        }
    }

    uchar* ptr = img.scanLine(0);
    ptr[0] = 0;
    ptr[1] = 0;
    ptr[2] = 0;
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

    for(int z = 0; z < sz; z++){
        uchar* ptr = img.scanLine(z);
        for(int y = 0; y< sy*3; ){
            QColor rgb = Helper::pseudoColor(buffer[z*sx*sy + y*sx + pos/3],mn,mx);
            ptr[y++] = rgb.red();
            ptr[y++] = rgb.green();
            ptr[y++] = rgb.blue();
        }
    }

    uchar* ptr = img.scanLine(0);
    ptr[0] = 0;
    ptr[1] = 0;
    ptr[2] = 0;
    return img;
}
