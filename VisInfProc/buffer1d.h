#ifndef FILTER1D_H
#define FILTER1D_H

#include <QVector>

/**
 * @brief The Filter1D class stores the filter coefficients for a 1d filter.
 */
class Buffer1D
{
public:
    Buffer1D();
    Buffer1D(int size);
    Buffer1D(const Buffer1D& f);

    Buffer1D& operator=(const Buffer1D &other);
    double& operator()(int i);
    double operator()(int i) const;

    int getSize() const{
        return size;
    }

    double* getBuff() const{
        return buffer;
    }

private:
    double* buffer;
    int size;
};

#endif // FILTER1D_H
