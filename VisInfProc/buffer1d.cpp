#include "buffer1d.h"
#include <assert.h>

Buffer1D::Buffer1D()
{
    size = 0;
    buffer = NULL;
}

Buffer1D::Buffer1D(int size)
{
    this->size = size;
    buffer = new double[size];
}

Buffer1D::Buffer1D(const Buffer1D& f){
    size = f.getSize();
    buffer = new double[size];
    memcpy(buffer,f.getBuff(),size*sizeof(double));
}

Buffer1D& Buffer1D::operator=(const Buffer1D &other)
{
    size = other.getSize();
    buffer = new double[size];
    memcpy(buffer,other.getBuff(),size*sizeof(double));
    return *this;
}

double& Buffer1D::operator()(int i)
{
    assert(i >= 0 && i < size);
    return buffer[i];
}
double Buffer1D::operator()(int i) const
{
    assert(i >= 0 && i < size);
    return buffer[i];
}
