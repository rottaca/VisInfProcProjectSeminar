#include "filter1d.h"

Filter1D::Filter1D(int size)
{
    this->size = size;
    v.resize(size);
}

Filter1D::Filter1D(const Filter1D& f){
    size = f.getSize();
    v = QVector<float>(f.getV());
}

float& Filter1D::operator()(int i)
{
    return v[i];
}
float Filter1D::operator()(int i) const
{
    return v[i];
}
