#include "filter2d.h"
#include <QString>

Filter2D::Filter2D(int sx, int sy)
{
    this->sx = sx;
    this->sy = sy;

    size_t s = sx*sy;
    qDebug(QString("Constructing 2D-Filter: %1 kB").arg(s/1024.0f*sizeof(float)).toLocal8Bit());
    buffer = new float[s];
    memset(buffer,0,s*sizeof(float));
}
Filter2D::Filter2D(const Filter2D& f)
{
    sx = f.getSizeX();
    sy = f.getSizeY();

    size_t s = sx*sy;
    buffer = new float[s];
    memcpy(buffer,f.getBuff(),s*sizeof(float));
}

float& Filter2D::operator()(int x, int y)
{
    return buffer[y*sx + x];
}
float Filter2D::operator()(int x, int y) const
{
    return buffer[y*sx + x];
}

Filter2D::~Filter2D()
{
    if(buffer != NULL)
        delete[] buffer;
    buffer = NULL;
}

QImage Filter2D::toImage() const
{
    QImage img(sx,sy,QImage::Format_Grayscale8);
    float mx = *std::max_element(buffer,buffer+sx*sy);
    float mn = *std::min_element(buffer,buffer+sx*sy);
    qDebug(QString("Min: %1 Max: %2").arg(mn).arg(mx).toLocal8Bit());

    for(int y = 0; y < sy; y++){
        uchar* ptr = img.scanLine(y);
        for(int x = 0; x < sx; x++){
            ptr[x] = qRound((buffer[y*sx+x]-mn)/mx*255.f);
        }
    }

    return img;
}
