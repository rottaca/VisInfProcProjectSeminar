#ifndef FILTER2D_H
#define FILTER2D_H

#include <QVector2D>
#include <QImage>

/**
 * @brief The Filter2D class stores the filter coefficients for a 2d filter.
 * The memory layout is x, then y
 */
class Buffer2D
{
public:
    Buffer2D();
    Buffer2D(int sx, int sy);
    Buffer2D(const Buffer2D& f);
    ~Buffer2D();

    Buffer2D &operator=(const Buffer2D& other);
    double& operator()(int x, int y);
    double operator()(int x, int y) const;

    int getSizeX() const{
        return sx;
    }
    int getSizeY() const{
        return sy;
    }
    double* getBuff() const{
        return buffer;
    }

    void resize(int sx, int sy);
    QImage toImage(double min = 0, double max = 0) const;

private:
    double* buffer;
    int sx,sy;
};

#endif // FILTER2D_H
