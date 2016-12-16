#include "convolution3d.h"
#include <QtMath>

Convolution3D::Convolution3D()
{
    readIdx = 0;
    writeIdx = 0;
}


Convolution3D::Convolution3D(int sx, int sy, int sz)
{
    buffer = Buffer3D(sx,sy,sz);
    readIdx = 0;
    writeIdx = 0;
}

void Convolution3D::convolute3D(Buffer3D &filter,QVector2D pos)
{
    int fx_2 = qFloor(filter.getSizeX()/2.0f);
    int fy_2 = qFloor(filter.getSizeY()/2.0f);

    // For every time slice
    for(int z = 0; z < filter.getSizeZ();z++){
        if(z >= buffer.getSizeZ())
            break;

        // Flip time axis
        float* ptrFilter = filter.getBuff();
        ptrFilter +=(filter.getSizeZ()-z-1)*filter.getSizeX()*filter.getSizeY();
        // Get time slice in the ring buffer
        float* ptrBuffer = buffer.getBuff();
        ptrBuffer += ((writeIdx + z) % buffer.getSizeZ())*buffer.getSizeX()*buffer.getSizeY();
        // for every entry in 2d grid
        for(int y = 0; y < filter.getSizeY(); y++){
            int yBuff = y - fy_2 + pos.y();
            // Skip invalid positions
            if(yBuff < 0 || yBuff >= buffer.getSizeY())
                continue;
            for(int x = 0; x < filter.getSizeX(); x++){
                int xBuff = x - fx_2 + pos.x();
                // Skip invalid positions
                if(xBuff < 0 || xBuff >= buffer.getSizeX())
                    continue;
                int idxBuff = yBuff*buffer.getSizeX() + xBuff;
                // Flip x and y directions in filter
                int idxFilter = (filter.getSizeY()-1 - y)*filter.getSizeX() + filter.getSizeX()-1-x;
                ptrBuffer[idxBuff] += ptrFilter[idxFilter];
            }
        }
    }
}

void Convolution3D::nextTimeSlot(Buffer2D* output)
{
    if(output != NULL){
        output->resize(buffer.getSizeX(),buffer.getSizeY());

        float* ptrOutput = output->getBuff();
        float* ptrBuffer = &buffer(0,0,writeIdx);
        for(int i = 0; i < buffer.getSizeY()*buffer.getSizeX(); i++){
            ptrOutput[i] = ptrBuffer[i];
        }
    }
    // Increase write and read pointer
    writeIdx = (writeIdx+1) % buffer.getSizeZ();
    readIdx = (readIdx+1) % buffer.getSizeZ();
}
