#include "buffer1d.h"

Buffer1D::Buffer1D()
{
    size = 0;
    v.resize(0);
}

Buffer1D::Buffer1D(int size)
{
    this->size = size;
    v.resize(size);
}

Buffer1D::Buffer1D(const Buffer1D& f){
    size = f.getSize();
    v = QVector<float>(f.getV());
}

float& Buffer1D::operator()(int i)
{
    return v[i];
}
float Buffer1D::operator()(int i) const
{
    return v[i];
}
