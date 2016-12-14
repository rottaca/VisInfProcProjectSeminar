#include "filter3d.h"

#include <QString>

Filter3D::Filter3D(int sx, int sy, int sz)
{
    this->sx = sx;
    this->sy = sy;
    this->sz = sz;

    size_t s = sx*sy*sz;
    qDebug(QString("Constructing 3D-Filter: %1 kB").arg(s/1024.0f*sizeof(float)).toLocal8Bit());
    buffer = new float[s];
    memset(buffer,0,s*sizeof(float));
}

float& Filter3D::operator()(int x, int y, int z)
{
    return buffer[z*sx*sy + y*sx + x];
}

float Filter3D::operator()(int x, int y, int z) const
{
    return buffer[z*sx*sy + y*sx + x];
}

Filter3D::~Filter3D()
{
    if(buffer != NULL)
        delete[] buffer;
    buffer = NULL;
}
