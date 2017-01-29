#ifndef FILTER1D_H
#define FILTER1D_H

#include "basebuffer.h"

/**
 * @brief The Filter1D class stores the filter coefficients for a 1d filter.
 */
class Buffer1D : public BaseBuffer
{
public:
    Buffer1D();
    Buffer1D(int size);
    Buffer1D(const Buffer1D& other);

    Buffer1D& operator=(const Buffer1D &other);

    float& operator()(int i);

    int getSize() const
    {
        return itemCnt;
    }

};

#endif // FILTER1D_H
