#include "buffer2d.h"
#include <QString>
#include <QRgb>
#include <QColor>
#include "helper.h"
#include <assert.h>

Buffer2D::Buffer2D()
{
    sx = sy = 0;
    buffer = NULL;
}

Buffer2D::Buffer2D(int sx, int sy)
{
    this->sx = sx;
    this->sy = sy;

    size_t s = sx*sy;
    buffer = new double[s];
    memset(buffer,0,s*sizeof(double));
}
Buffer2D::Buffer2D(const Buffer2D& f)
{
    sx = f.getSizeX();
    sy = f.getSizeY();

    size_t s = sx*sy;
    if(s <= 0)
    {
        buffer = NULL;
        return;
    }
    buffer = new double[s];
    memcpy(buffer,f.getBuff(),s*sizeof(double));
}

Buffer2D &Buffer2D::operator=(const Buffer2D& other)
{
    if(this != &other){
        resize(other.getSizeX(),other.getSizeY());
        memcpy(buffer,other.getBuff(),sx*sy*sizeof(double));
    }

    return *this;
}
double& Buffer2D::operator()(int x, int y)
{
    assert(x >= 0 && x < sx);
    assert(y >= 0 && y < sy);
    return buffer[y*sx + x];
}
double Buffer2D::operator()(int x, int y) const
{
    assert(x >= 0 && x < sx);
    assert(y >= 0 && y < sy);
    return buffer[y*sx + x];
}

Buffer2D::~Buffer2D()
{
    if(buffer != NULL)
        delete[] buffer;
    buffer = NULL;
}
void Buffer2D::resize(int sx, int sy)
{
    if(sx == this->sx && sy == this->sy)
        return;

    if(buffer != NULL)
        delete[] buffer;

    this->sx = sx;
    this->sy = sy;
    size_t s = sx*sy;
    buffer = new double[s];
}

QImage Buffer2D::toImage(double min, double max) const
{
    QImage img(sx,sy,QImage::Format_RGB888);
    double mx = max;
    double mn = min;
    if(min == 0 && max == 0){
        mx = *std::max_element(buffer,buffer+sx*sy);
        mn = *std::min_element(buffer,buffer+sx*sy);
    }
    for(int y = 0; y < sy; y++){
        uchar* ptr = img.scanLine(y);
        for(int x = 0; x < sx*3; ){
            QColor rgb = Helper::pseudoColor(buffer[y*sx+x/3],mn,mx);
            ptr[x++] = rgb.red();
            ptr[x++] = rgb.green();
            ptr[x++] = rgb.blue();
        }
    }

    return img;
}


