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
    currentWindowStartTime = 0;
    startTime = -1;

    fset = new FilterSet[orientations.length()];
    conv = new Convolution3D[orientations.length()*4];
    motionRight = new Buffer2D[orientations.length()];
    motionLeft = new Buffer2D[orientations.length()];
    for(int i = 0; i < orientations.length(); i++){
        fset[i] = FilterSet(fs,orientations.at(i));
        for(int j = 0; j < 4; j++){
            conv[i*4+j] = Convolution3D(128,128,fset[i].spatialTemporal[FilterSet::LEFT1].getSizeZ());
        }
    }
    // Time per timeslot
    timeRes =(float)fs.timewindow_us/fs.temporalSteps;
    isMotionEnergyReady = false;
}

MotionEnergyEstimator::~MotionEnergyEstimator()
{
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

    int deltaT = (e.timestamp-startTime) - currentWindowStartTime;
    // Do we have to skip any timeslots ? Is the new event too new for the current slot ?
    int timeSlotsToSkip = qFloor((float)deltaT/timeRes);

    for(int i = 0; i < orientations.length(); i++){

        // Skip time slots
        // TODO Skip everything ?
        //qDebug("ProcessEvent");
        if(timeSlotsToSkip > 0){
            Buffer2D left1,left2,right1,right2;
            conv[i*4+0].nextTimeSlot(&left1,timeSlotsToSkip);
            conv[i*4+1].nextTimeSlot(&left2,timeSlotsToSkip);
            conv[i*4+2].nextTimeSlot(&right1,timeSlotsToSkip);
            conv[i*4+3].nextTimeSlot(&right2,timeSlotsToSkip);

            computeMotionEnergy(left1,left2,motionLeft[i]);
            computeMotionEnergy(right1,right2,motionRight[i]);
            isMotionEnergyReady = true;

            currentWindowStartTime += timeRes*timeSlotsToSkip;

            while(timeWindowEvents.size() > 0
                  && timeWindowEvents.front().timestamp < currentWindowStartTime)
                timeWindowEvents.pop_front();
        }
        // Convolute all four filters for this direction
        conv[i*4+0].convolute3D(fset[i].spatialTemporal[FilterSet::LEFT1],ePos);
        conv[i*4+1].convolute3D(fset[i].spatialTemporal[FilterSet::LEFT2],ePos);
        conv[i*4+2].convolute3D(fset[i].spatialTemporal[FilterSet::RIGHT1],ePos);
        conv[i*4+3].convolute3D(fset[i].spatialTemporal[FilterSet::RIGHT2],ePos);
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

