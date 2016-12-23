#ifndef OPTICFLOWESTIMATOR_H
#define OPTICFLOWESTIMATOR_H

#include "motionenergyestimator.h"
#include "dvseventhandler.h"

#include <QList>
#include <assert.h>
#include <QVector>

class OpticFlowEstimator
{
public:
    OpticFlowEstimator(QList<FilterSettings> settings, QList<float> orientations);
    ~OpticFlowEstimator();

    void processEvent(DVSEventHandler::DVSEvent event);

    void setEnergyThreshold(int filterNr, double threshold){
        assert(filterNr < energyEstimatorCnt);
        filterThresholds[filterNr] = threshold;
    }

    long getMotionEnergy(int filterNr, int orientationIdx, Buffer2D &energyLeft, Buffer2D &energyRight){
        assert(filterNr < energyEstimatorCnt);
        assert(orientationIdx < orientations.length());
        energyLeft = motionEnergiesLeft[filterNr*orientations.length() + orientationIdx];
        energyRight = motionEnergiesRight[filterNr*orientations.length() + orientationIdx];
        return updateTimeStamps[filterNr];
    }

    QVector<DVSEventHandler::DVSEvent> getEventsInWindow(int filterNr){
        assert(filterNr < energyEstimatorCnt);
        return motionEnergyEstimators[filterNr]->getEventsInWindow();
    }


private:
    int energyEstimatorCnt;
    MotionEnergyEstimator **motionEnergyEstimators;
    Buffer2D *motionEnergiesLeft,*motionEnergiesRight;
    long *updateTimeStamps;
    double *filterThresholds;
    QList<float> orientations;
    QList<FilterSettings> settings;
};

#endif // OPTICFLOWESTIMATOR_H
