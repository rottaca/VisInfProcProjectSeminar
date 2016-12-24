#include "motionenergyestimator.h"
#include <QtMath>
#include <QElapsedTimer>
#include <QPainter>
#include <assert.h>
#include <iostream>

MotionEnergyEstimator::MotionEnergyEstimator(FilterSettings fs, QList<float> orientations)
{
    this->fsettings = fs;
    this->orientations = orientations;
    currentSlotStartTime = 0;
    startTime = -1;

    fset = new FilterSet*[orientations.length()];
    conv = new Convolution3D*[orientations.length()*4];
    motionRight = new Buffer2D[orientations.length()];
    motionLeft = new Buffer2D[orientations.length()];
    for(int i = 0; i < orientations.length(); i++){
        fset[i] = new FilterSet(fs,orientations.at(i));
        for(int j = 0; j < 4; j++){
            conv[i*4+j] = new Convolution3D(128,128,fset[i]->sz);
        }
    }
    // Time per timeslot
    timeRes =(float)fs.timewindow_us/fs.temporalSteps;
    isMotionEnergyReady = false;
}

MotionEnergyEstimator::~MotionEnergyEstimator()
{
    for(int i = 0; i < orientations.length(); i++){
        delete fset[i];
        for(int j = 0; j < 4; j++){
            delete conv[i*4+j];
        }
    }

    delete[] fset;
    fset = NULL;
    delete[] fset;
    conv = NULL;
    delete[] motionRight;
    motionRight = NULL;
    delete[] motionLeft;
    motionLeft = NULL;
}
void MotionEnergyEstimator::processEvent(DVSEventHandler::DVSEvent e)
{
    if(startTime == -1)
        startTime = e.timestamp;

    if(e.On)
        return;

    // Events are sorted by time
    assert(timeWindowEvents.size() == 0 || e.timestamp >= timeWindowEvents.back().timestamp);

    timeWindowEvents.push_back(e);
    QVector2D ePos(e.posX,e.posY);

    int deltaT = (e.timestamp-startTime) - currentSlotStartTime;
    // Do we have to skip any timeslots ? Is the new event too new for the current slot ?
    int timeSlotsToSkip = qFloor((float)deltaT/timeRes);

    for(int i = 0; i < orientations.length(); i++){

        // Skip time slots
        if(timeSlotsToSkip > 0){
            Buffer2D left1,left2,right1,right2;
            conv[i*4+0]->nextTimeSlot(&left1,timeSlotsToSkip);
            conv[i*4+1]->nextTimeSlot(&left2,timeSlotsToSkip);
            conv[i*4+2]->nextTimeSlot(&right1,timeSlotsToSkip);
            conv[i*4+3]->nextTimeSlot(&right2,timeSlotsToSkip);

            computeMotionEnergy(left1,left2,motionLeft[i]);
            computeMotionEnergy(right1,right2,motionRight[i]);
            isMotionEnergyReady = true;

            currentSlotStartTime += timeRes*timeSlotsToSkip;

            while(timeWindowEvents.size() > 0
                  && timeWindowEvents.front().timestamp - startTime < currentSlotStartTime - fsettings.timewindow_us)
                timeWindowEvents.pop_front();
        }
        // Convolute all four filters for this direction
        long fs_x = fset[i]->sx;
        long fs_y = fset[i]->sy;
        long fs_z = fset[i]->sz;
        conv[i*4+0]->convolute3D(fset[i]->gpuSpatialTemporal[FilterSet::LEFT1],fs_x,fs_y,fs_z,ePos);
        conv[i*4+1]->convolute3D(fset[i]->gpuSpatialTemporal[FilterSet::LEFT2],fs_x,fs_y,fs_z,ePos);
        conv[i*4+2]->convolute3D(fset[i]->gpuSpatialTemporal[FilterSet::RIGHT1],fs_x,fs_y,fs_z,ePos);
        conv[i*4+3]->convolute3D(fset[i]->gpuSpatialTemporal[FilterSet::RIGHT2],fs_x,fs_y,fs_z,ePos);
    }
}

void MotionEnergyEstimator::computeMotionEnergy(Buffer2D &one, Buffer2D &two, Buffer2D &energy)
{
    double* ptrOne = one.getBuff();
    double* ptrTwo = two.getBuff();
    energy.resize(one.getSizeX(),one.getSizeY());
    double* ptrEnergy = energy.getBuff();

    for(int i = 0; i < one.getSizeX()*one.getSizeY(); i++){
        ptrEnergy[i] = qSqrt(ptrOne[i]*ptrOne[i] + ptrTwo[i]*ptrTwo[i]);
    }
}

