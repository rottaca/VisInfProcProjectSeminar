#ifndef FILTER2D_H
#define FILTER2D_H

#include <QImage>

#include "basebuffer.h"

#include <cuda.h>
#include <cuda_runtime.h>

extern void cuda2DBufferToRGBImage(int sx, int sy,float min, float max,
                                   float* gpuBuffer, unsigned char* gpuImage,cudaStream_t cudaStream);

/**
 * @brief The Filter2D class stores the filter coefficients for a 2d filter.
 * The memory layout is x, then y
 */
class Buffer2D : public BaseBuffer
{
public:
    Buffer2D();
    Buffer2D(int sx, int sy);
    Buffer2D(const Buffer2D& other);

    Buffer2D &operator=(const Buffer2D& other);
    float& operator()(int x, int y);

    int getSizeX() const
    {
        return sx;
    }
    int getSizeY() const
    {
        return sy;
    }

    void resize(int sx, int sy);
    QImage toImage(float min = 0, float max = 0) const;

private:
    int sx,sy;
};

#endif // FILTER2D_H
