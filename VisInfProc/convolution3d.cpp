#include "convolution3d.h"
#include <QtMath>
#include <assert.h>
#include <QColor>
#include "helper.h"

Convolution3D::Convolution3D()
{
    writeIdx = 0;
}


Convolution3D::Convolution3D(int sx, int sy, int sz)
{
    buffer = Buffer3D(sx,sy,sz);
    writeIdx = 0;
}

void Convolution3D::convolute3D(Buffer3D &filter, QVector2D pos)
{
    // TODO SPEEDUP!
    // Precompute xBuff, yBuff, idxBuff, idxFilter where possible
    int fs_x = filter.getSizeX();
    int fs_y = filter.getSizeY();
    int fx_2 = qFloor(fs_x/2.0f);
    int fy_2 = qFloor(fs_y/2.0f);
    int fs_xy = fs_x*fs_y;
    int fs_z = filter.getSizeZ();

    int bs_x = buffer.getSizeX();
    int bs_y = buffer.getSizeY();
    int bs_xy = bs_x*bs_y;
    int bs_z = buffer.getSizeZ();

    // For every time slice
#pragma omp parallel for
    for(int z = 0; z < fs_z;z++){
        if(z >= bs_z)
            continue;

        // Get Pointer to time slice in 3d buffer
        double* ptrFilterTSlice = filter.getBuff() + z*fs_xy;
        // Get time slice in the ring buffer (flip z coordinate)
        double* ptrBufferTSlice = buffer.getBuff() + ((writeIdx + (fs_z - 1) - z ) % bs_z)*bs_xy;

        // for every entry in 2d grid
        // Flip x and y directions in buffer
        for(int yFilter = 0; yFilter < fs_y; yFilter++){

            int yBuff = (fs_y - 1 - yFilter) - fy_2 + pos.y();
            // Skip invalid positions
            if(yBuff < 0 || yBuff >= bs_y)
                continue;

            for(int xFilter = 0; xFilter < fs_x; xFilter++){

                int xBuff = (fs_x - 1 - xFilter) - fx_2 + pos.x();
                // Skip invalid positions
                if(xBuff < 0 || xBuff >= bs_x)
                    continue;

                int idxBuff = yBuff*bs_x + xBuff;
                int idxFilter = yFilter *fs_x + xFilter;
                ptrBufferTSlice[idxBuff] += ptrFilterTSlice[idxFilter];
            }
        }
    }
}

void Convolution3D::nextTimeSlot(Buffer2D* output, int slotsToSkip)
{
    double* ptrOutput= NULL;
    double* ptrBuffer = &buffer(0,0,writeIdx);

    if(output != NULL){
        output->resize(buffer.getSizeX(),buffer.getSizeY());
        ptrOutput = output->getBuff();
    }

    for(int i = 0; i < buffer.getSizeY()*buffer.getSizeX(); i++){
        ptrOutput[i] = ptrBuffer[i];
    }
    long pageSize = buffer.getSizeX()*buffer.getSizeY()*sizeof(double);

    if(writeIdx+slotsToSkip >= buffer.getSizeZ()){
        long slotCntOverflow = (writeIdx+slotsToSkip) % buffer.getSizeZ();
        memset(&buffer(0,0,0),0,pageSize*slotCntOverflow);

        slotsToSkip-=slotCntOverflow;
    }
    // Skip at the end of the buffer
    memset(ptrBuffer,0,pageSize*slotsToSkip);

    // Increase write and read pointer
    writeIdx = (writeIdx+slotsToSkip) % buffer.getSizeZ();
}
QImage Convolution3D::toOrderedImageXZ(int orderStart, int slicePos, float min, float max)
{
    int sx = buffer.getSizeX();
    int sy = buffer.getSizeY();
    int sz = buffer.getSizeZ();
    assert(slicePos >= 0 && slicePos < sy);
    assert(orderStart >= 0 && orderStart < sz);
    QImage img(sx,sz,QImage::Format_RGB888);
    float mx = max;
    float mn = min;
    double* buff = buffer.getBuff();
    if(min == 0 && max == 0){
        mx = *std::max_element(buff,buff+sz*sx*sy);
        mn = *std::min_element(buff,buff+sz*sx*sy);
    }

    for(int z = 0; z < sz; z++){
        int zWarped = (orderStart + z) % sz;
        uchar* ptr = img.scanLine(z);
        for(int x = 0; x < sx*3; ){
            QColor rgb = Helper::pseudoColor(buff[zWarped*sx*sy + slicePos*sx + x/3],mn,mx);
            ptr[x++] = rgb.red();
            ptr[x++] = rgb.green();
            ptr[x++] = rgb.blue();
        }
    }

    uchar* ptr = img.scanLine(0);
    ptr[0] = 0;
    ptr[1] = 0;
    ptr[2] = 0;
    return img;
}

QImage Convolution3D::toOrderedImageYZ(int orderStart, int slicePos, float min, float max)
{
    int sx = buffer.getSizeX();
    int sy = buffer.getSizeY();
    int sz = buffer.getSizeZ();
    assert(slicePos >= 0 && slicePos < sx);
    assert(orderStart >= 0 && orderStart < sz);
    QImage img(sy,sz,QImage::Format_RGB888);
    float mx = max;
    float mn = min;
    double* buff = buffer.getBuff();
    if(min == 0 && max == 0){
        mx = *std::max_element(buff,buff+sz*sx*sy);
        mn = *std::min_element(buff,buff+sz*sx*sy);
    }

    for(int z = 0; z < sz; z++){
        int zWarped = (orderStart + z) % sz;
        uchar* ptr = img.scanLine(z);
        for(int y = 0; y < sy*3; ){
            QColor rgb = Helper::pseudoColor(buff[zWarped*sx*sy + y/3*sx + slicePos],mn,mx);
            ptr[y++] = rgb.red();
            ptr[y++] = rgb.green();
            ptr[y++] = rgb.blue();
        }
    }

    uchar* ptr = img.scanLine(0);
    ptr[0] = 0;
    ptr[1] = 0;
    ptr[2] = 0;
    return img;
}
