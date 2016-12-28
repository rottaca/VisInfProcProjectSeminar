#ifndef FILTER3D_H
#define FILTER3D_H

#include <QImage>
#include "basebuffer.h"

/**
 * @brief The Filter3D class stores the filter coefficients for a 3d filter.
 * The memory layout is x, then y, then z
 */
class Buffer3D : public BaseBuffer
{
public:
    Buffer3D();
    Buffer3D(int sx, int sy, int sz);
    Buffer3D(const Buffer3D& other);

    Buffer3D &operator=(const Buffer3D& other);
    double& operator()(int x,int y,int z);

    int getSizeX() const{
        return sx;
    }
    int getSizeY() const{
        return sy;
    }
    int getSizeZ() const{
        return sz;
    }

    void resize(int sx, int sy, int sz);
    QImage toImageXY(int pos, double min = 0, double max = 0) const;
    QImage toImageXZ(int pos, double min = 0, double max = 0) const;
    QImage toImageYZ(int pos, double min = 0, double max = 0) const;

private:
    int sx,sy,sz;

};

#endif // FILTER3D_H
