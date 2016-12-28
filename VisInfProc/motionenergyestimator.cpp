#include "motionenergyestimator.h"
#include <QtMath>
#include <QElapsedTimer>
#include <QPainter>
#include <assert.h>
#include <iostream>
#include <QFile>

MotionEnergyEstimator::MotionEnergyEstimator(FilterSettings fs, QList<float> orientations)
{
    this->fsettings = fs;
    this->orientations = orientations;
    currentSlotStartTime = 0;
    startTime = -1;

    fset = new FilterSet*[orientations.length()];
    conv = new Convolution3D*[orientations.length()*4];
    opponentMotionEnergy = new Buffer2D[orientations.length()];
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
    delete[] opponentMotionEnergy;
    opponentMotionEnergy = NULL;
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
            conv[i*4+0]->nextTimeSlot(&left1,timeSlotsToSkip);
            conv[i*4+1]->nextTimeSlot(&left2,timeSlotsToSkip);
            conv[i*4+2]->nextTimeSlot(&right1,timeSlotsToSkip);
            conv[i*4+3]->nextTimeSlot(&right2,timeSlotsToSkip);

//            QFile file("left.png");
//            file.open(QIODevice::WriteOnly);
//            conv[i*4+0]->buffer.toImageXZ(64).save(&file,"PNG");

            computeMotionEnergy(left1,left2,energyL);
            computeMotionEnergy(right1,right2,energyR);

            opponentMotionEnergy[i] = energyR;
            opponentMotionEnergy[i] -= energyL;
            //opponentMotionEnergy[i] = left1;
            isMotionEnergyReady = true;

            currentSlotStartTime += timeRes*timeSlotsToSkip;

            while(timeWindowEvents.size() > 0
                  && timeWindowEvents.front().timestamp - startTime < currentSlotStartTime - fsettings.timewindow_us)
                timeWindowEvents.pop_front();
        }
        // Convolute all four filters for this direction
        conv[i*4+0]->convolute3D(fset[i]->spatialTemporal[FilterSet::LEFT1],ePos);
        conv[i*4+1]->convolute3D(fset[i]->spatialTemporal[FilterSet::LEFT2],ePos);
        conv[i*4+2]->convolute3D(fset[i]->spatialTemporal[FilterSet::RIGHT1],ePos);
        conv[i*4+3]->convolute3D(fset[i]->spatialTemporal[FilterSet::RIGHT2],ePos);

    }
}

void MotionEnergyEstimator::computeMotionEnergy(Buffer2D &one, Buffer2D &two, Buffer2D &energy)
{
    double* ptrOne = one.getCPUPtr();
    double* ptrTwo = two.getCPUPtr();
    energy.resize(one.getSizeX(),one.getSizeY());
    double* ptrEnergy = energy.getCPUPtr();

    for(int i = 0; i < one.getSizeX()*one.getSizeY(); i++){
        ptrEnergy[i] = qSqrt(ptrOne[i]*ptrOne[i] + ptrTwo[i]*ptrTwo[i]);
    }
}

