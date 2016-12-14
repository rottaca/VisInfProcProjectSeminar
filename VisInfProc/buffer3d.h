#ifndef FILTER3D_H
#define FILTER3D_H

#include <QVector3D>

/**
 * @brief The Filter3D class stores the filter coefficients for a 3d filter.
 * The memory layout is x, then y, then z
 */
class Buffer3D
{
public:
    Buffer3D();
    Buffer3D(int sx, int sy, int sz);
    Buffer3D(const Buffer3D& b);
    ~Buffer3D();

    Buffer3D &operator=(const Buffer3D& other);
    float& operator()(int x,int y,int z);
    float operator()(int x,int y,int z) const;
    Buffer3D operator-(Buffer3D& b) const;
    Buffer3D operator+(Buffer3D& b) const;

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

    QImage toImageXY(int pos);
    QImage toImageXZ(int pos);
    QImage toImageYZ(int pos);

private:
    float* buffer;
    int sx,sy,sz;

};

#endif // FILTER3D_H
