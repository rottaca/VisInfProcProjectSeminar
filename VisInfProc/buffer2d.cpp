#include "buffer2d.h"
#include <QString>
#include <QRgb>
#include <QColor>
#include "helper.h"

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
    qDebug(QString("Constructing 2D-Filter: %1 kB").arg(s/1024.0f*sizeof(float)).toLocal8Bit());
    buffer = new float[s];
    memset(buffer,0,s*sizeof(float));
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
    buffer = new float[s];
    memcpy(buffer,f.getBuff(),s*sizeof(float));
}

Buffer2D &Buffer2D::operator=(const Buffer2D& other)
{
    if(this != &other){
        if(buffer != NULL)
        {
            delete []buffer;
            buffer = NULL;
        }

        sx = other.getSizeX();
        sy = other.getSizeY();

        size_t s = sx*sy;
        buffer = new float[s];
        memcpy(buffer,other.getBuff(),s*sizeof(float));
    }

    return *this;
}
float& Buffer2D::operator()(int x, int y)
{
    return buffer[y*sx + x];
}
float Buffer2D::operator()(int x, int y) const
{
    return buffer[y*sx + x];
}

Buffer2D::~Buffer2D()
{
    if(buffer != NULL)
        delete[] buffer;
    buffer = NULL;
}

QImage Buffer2D::toImage() const
{
    QImage img(sx,sy,QImage::Format_RGB888);
    float mx = *std::max_element(buffer,buffer+sx*sy);
    float mn = *std::min_element(buffer,buffer+sx*sy);
    qDebug(QString("Min: %1 Max: %2").arg(mn).arg(mx).toLocal8Bit());

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


