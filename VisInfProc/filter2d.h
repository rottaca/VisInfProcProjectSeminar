#ifndef FILTER2D_H
#define FILTER2D_H

#include <QVector2D>
#include <QImage>

/**
 * @brief The Filter2D class stores the filter coefficients for a 2d filter.
 * The memory layout is x, then y
 */
class Filter2D
{
public:
    Filter2D(int sx, int sy);
    Filter2D(const Filter2D& f);
    ~Filter2D();

    float& operator()(int x, int y);
    float operator()(int x, int y) const;

    int getSizeX() const{
        return sx;
    }
    int getSizeY() const{
        return sy;
    }
    float* getBuff() const{
        return buffer;
    }
    QImage toImage() const;

private:
    float* buffer;
    int sx,sy;
};

#endif // FILTER2D_H
