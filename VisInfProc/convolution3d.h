#ifndef CONVOLUTION3D_H
#define CONVOLUTION3D_H

#include "buffer2d.h"
#include "buffer3d.h"
#include <QVector2D>

/**
 * @brief The Convolution3D class Datastructure to convolute spatial temporal 3d filters over time and space
 */
class Convolution3D
{

public:
    Convolution3D();
    Convolution3D(int sx, int sy, int sz);

    /**
     * @brief convolute3D Convolutes the given filter into a forward looking ring buffer
     * @param filter The spatial temporal filter is centered in the spatial dimensions and at t=0 in the temporal dimension
     * @param pos The spatial position where the filter is placed in the buffer at the current time slice.
     */
    void convolute3D(Buffer3D &filter, QVector2D pos);
    /**
     * @brief nextTimeSlot Returns the spatial filter response for the current time and increases the read and write indices
     * @param output
     */
    void nextTimeSlot(Buffer2D* output);

    Buffer3D *getBuff(){
        return &buffer;
    }

    int getReadIdx(){
        return readIdx;
    }

    int getWriteIdx(){
        return writeIdx;
    }

private:
    Buffer3D buffer;
    int readIdx,writeIdx;
};

#endif // CONVOLUTION3D_H
