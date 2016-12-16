#include "buffer1d.h"

Buffer1D::Buffer1D()
{
    size = 0;
    buffer = NULL;
}

Buffer1D::Buffer1D(int size)
{
    this->size = size;
    buffer = new float[size];
}

Buffer1D::Buffer1D(const Buffer1D& f){
    size = f.getSize();
    buffer = new float[size];
    memcpy(buffer,f.getBuff(),size*sizeof(float));
}

Buffer1D& Buffer1D::operator=(const Buffer1D &other)
{
    size = other.getSize();
    buffer = new float[size];
    memcpy(buffer,other.getBuff(),size*sizeof(float));
    return *this;
}

float& Buffer1D::operator()(int i)
{
    return buffer[i];
}
float Buffer1D::operator()(int i) const
{
    return buffer[i];
}
