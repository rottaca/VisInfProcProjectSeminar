#include "convolution3d.h"
#include <QtMath>
#include <assert.h>
#include <QColor>
#include "helper.h"
#include "cuda_helper.h"
#include "cuda_convolution3d.h"

Convolution3D::Convolution3D()
{
    writeIdx = 0;
    gpuBuffer = NULL;
}

Convolution3D::Convolution3D(int sx, int sy, int sz)
{
    buffer = Buffer3D(sx,sy,sz);
    writeIdx = 0;

    gpuBuffer = cudaCreateDoubleBuffer(sx*sy*sz);
    cudaSetDoubleBuffer(gpuBuffer,0,sx*sy*sz);
}
Convolution3D::~Convolution3D()
{
    cudaFreeDoubleBuffer(gpuBuffer);
    gpuBuffer = NULL;
}

void Convolution3D::convolute3D(double* gpuFilter, int fs_x,int fs_y, int fs_z, QVector2D pos)
{
    int bs_x = buffer.getSizeX();
    int bs_y = buffer.getSizeY();
    int bs_z = buffer.getSizeZ();

    cudaConvolution3D(gpuBuffer,writeIdx,bs_x,bs_y,bs_z,
                      gpuFilter,fs_x,fs_y,fs_z,
                      pos.x(),pos.y());
    return;
}

void Convolution3D::nextTimeSlot(Buffer2D* output, int slotsToSkip)
{
    long pageSize = buffer.getSizeX()*buffer.getSizeY();

    if(output != NULL){
        output->resize(buffer.getSizeX(),buffer.getSizeY());

        cudaDownloadDoubleBuffer(gpuBuffer + pageSize*writeIdx,
                                 output->getBuff(),
                                 pageSize);
    }

    if(writeIdx+slotsToSkip > buffer.getSizeZ())
    {
        long slotCntOverflow = (writeIdx+slotsToSkip) % buffer.getSizeZ();
        // clear at the beginning of the buffer
        cudaSetDoubleBuffer(gpuBuffer,0,pageSize*slotCntOverflow);
        slotsToSkip-=slotCntOverflow;
    }
    if(slotsToSkip > 0){
        // Clear from current ring buffer position
        cudaSetDoubleBuffer(gpuBuffer + writeIdx*pageSize,0,pageSize*slotsToSkip);
    }
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
