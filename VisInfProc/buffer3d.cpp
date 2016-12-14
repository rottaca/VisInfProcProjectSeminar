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
    buffer = new float[s];
    memset(buffer,0,s*sizeof(float));
}
Buffer3D::Buffer3D(const Buffer3D& b)
{
    sx = b.getSizeX();
    sy = b.getSizeY();
    sz = b.getSizeZ();

    size_t s = sx*sy*sz;
    buffer = new float[s];
    memcpy(buffer,b.getBuff(),s*sizeof(float));
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
        buffer = new float[s];
        memcpy(buffer,other.getBuff(),s*sizeof(float));
    }

    return *this;
}

float& Buffer3D::operator()(int x, int y, int z)
{
    return buffer[z*sx*sy + y*sx + x];
}

float Buffer3D::operator()(int x, int y, int z) const
{
    return buffer[z*sx*sy + y*sx + x];
}

Buffer3D Buffer3D::operator-(Buffer3D& b) const
{
    assert(sx == b.getSizeX() && sy == b.getSizeY() && sz == b.getSizeZ());
    Buffer3D buff(sx,sy,sz);
    float *ptrRes = buff.getBuff();
    float *ptrB = b.getBuff();
    for(int i = 0; i < sx*sy*sz; i++){
        ptrRes[i] = buffer[i]-ptrB[i];
    }

    return buff;
}

Buffer3D Buffer3D::operator+(Buffer3D& b) const
{
    assert(sx == b.getSizeX() && sy == b.getSizeY() && sz == b.getSizeZ());
    Buffer3D buff(sx,sy,sz);
    float *ptrRes = buff.getBuff();
    float *ptrB = b.getBuff();
    for(int i = 0; i < sx*sy*sz; i++){
        ptrRes[i] = buffer[i]-ptrB[i];
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

QImage Buffer3D::toImageXY(int pos)
{
    assert(pos >= 0 && pos < sz);
    QImage img(sx,sy,QImage::Format_RGB888);
    float mx = *std::max_element(buffer+pos*sx*sy,buffer+sx*sy+pos*sx*sy);
    float mn = *std::min_element(buffer+pos*sx*sy,buffer+sx*sy+pos*sx*sy);

    float* buffPtr = buffer+pos*sx*sy;

    for(int y = 0; y < sy; y++){
        uchar* ptr = img.scanLine(y);
        for(int x = 0; x < sx*3; ){
            QColor rgb = Helper::pseudoColor(buffPtr[y*sx+x/3],mn,mx);
            ptr[x++] = rgb.red();
            ptr[x++] = rgb.green();
            ptr[x++] = rgb.blue();
        }
    }

    return img;
}
QImage Buffer3D::toImageXZ(int pos)
{
    assert(pos >= 0 && pos < sy);
    QImage img(sx,sz,QImage::Format_RGB888);
    float mx = buffer[0];
    float mn = buffer[0];

    for(int z = 0; z < sz; z++){
        for(int x = 0; x < sx; x++){
            float v = buffer[z*sx*sy + pos*sx + x];
            if(mx < v)
                mx = v;
            if(mn > v)
                mn = v;
        }
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

    return img;
}
QImage Buffer3D::toImageYZ(int pos)
{
    assert(pos >= 0 && pos < sx);
    QImage img(sy,sz,QImage::Format_RGB888);
    float mx = buffer[0];
    float mn = buffer[0];

    for(int z = 0; z < sz; z++){
        for(int y = 0; y< sy; y++){
            float v = buffer[z*sx*sy + y*sx + pos];
            if(mx < v)
                mx = v;
            if(mn > v)
                mn = v;
        }
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

    return img;
}
