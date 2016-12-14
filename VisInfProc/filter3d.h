#ifndef FILTER3D_H
#define FILTER3D_H

#include <QVector3D>

/**
 * @brief The Filter3D class stores the filter coefficients for a 3d filter.
 * The memory layout is x, then y, then z
 */
class Filter3D
{
public:
    Filter3D(int sx, int sy, int sz);
    ~Filter3D();

    float& operator()(int x,int y,int z);
    float operator()(int x,int y,int z) const;

    int getSizeX() const{
        return sx;
    }
    int getSizeY() const{
        return sy;
    }
    int getSizeZ() const{
        return sz;
    }
    float* getBuff() const{
        return buffer;
    }

private:
    float* buffer;
    int sx,sy,sz;

};

#endif // FILTER3D_H
