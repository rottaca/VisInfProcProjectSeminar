#ifndef MOTIONENERGYESTIMATOR_H
#define MOTIONENERGYESTIMATOR_H

#include "buffer1d.h"
#include "buffer2d.h"
#include "buffer3d.h"
#include "filtermanager.h"
#include "filtersettings.h"
#include "filterset.h"
#include "convolution3d.h"
#include "dvseventhandler.h"

#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>

extern void cudaComputeOpponentMotionEnergy(int sx, int sy,
                                    double* gpul1,double* gpul2,
                                    double* gpur1,double* gpur2,
                                   double* gpuEnergy);

class MotionEnergyEstimator
{
public:
    MotionEnergyEstimator(FilterSettings fs, QList<float> orientations);
    ~MotionEnergyEstimator();

    FilterSettings getSettings(){
        return fsettings;
    }

    void processEvent(DVSEventHandler::DVSEvent e);

    bool isEnergyReady(){
        return isMotionEnergyReady;
    }

    void getMotionEnergy(int orientationIdx,Buffer2D &oppMoEnergy)
    {
        assert(orientationIdx < orientations.length());
        oppMoEnergy = opponentMotionEnergy[orientationIdx];
        isMotionEnergyReady = false;
    }

    QVector<DVSEventHandler::DVSEvent> getEventsInWindow(){
        return timeWindowEvents;
    }

    long getWindowStartTime(){
        return currentSlotStartTime;
    }


private:
    FilterSettings fsettings;
    FilterSet** fset;
    Convolution3D** conv;
    QList<float> orientations;
    long currentSlotStartTime;
    int startTime;
    float timeRes;
    QVector<DVSEventHandler::DVSEvent> timeWindowEvents;

    Buffer2D left1,left2,right1,right2;
    Buffer2D energyL,energyR;
    bool isMotionEnergyReady;
    Buffer2D *opponentMotionEnergy;

};

#endif // MOTIONENERGYESTIMATOR_H
