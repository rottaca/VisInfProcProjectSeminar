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

    void getMotionEnergy(int orientationIdx,Buffer2D &left, Buffer2D &right)
    {
        assert(orientationIdx < orientations.length());
        left = motionLeft[orientationIdx];
        right = motionRight[orientationIdx];
        isMotionEnergyReady = false;
    }

    QVector<DVSEventHandler::DVSEvent> getEventsInWindow(){
        return timeWindowEvents;
    }

    long getWindowStartTime(){
        return currentSlotStartTime;
    }


private:
    void computeMotionEnergy(Buffer2D &one, Buffer2D &two, Buffer2D &energy);

private:
    FilterSettings fsettings;
    FilterSet** fset;
    Convolution3D** conv;
    QList<float> orientations;
    long currentSlotStartTime;
    int startTime;
    float timeRes;
    QVector<DVSEventHandler::DVSEvent> timeWindowEvents;

    bool isMotionEnergyReady;
    Buffer2D *motionRight;
    Buffer2D *motionLeft;

};

#endif // MOTIONENERGYESTIMATOR_H
