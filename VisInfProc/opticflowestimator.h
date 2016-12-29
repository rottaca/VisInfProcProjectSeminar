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
        assert(filterNr >= 0);
        assert(filterNr < energyEstimatorCnt);
        filterThresholds[filterNr] = threshold;
    }

    long getMotionEnergy(int filterNr, int orientationIdx, Buffer2D &opponentMotionEnergy){
        assert(filterNr >= 0);
        assert(filterNr < energyEstimatorCnt);
        assert(orientationIdx >= 0);
        assert(orientationIdx < orientations.length());
        opponentMotionEnergy = opponentMotionEnergies[filterNr*orientations.length() + orientationIdx];
        return updateTimeStamps[filterNr];
    }

    QVector<DVSEventHandler::DVSEvent> getEventsInWindow(int filterNr){
        assert(filterNr >= 0);
        assert(filterNr < energyEstimatorCnt);
        return motionEnergyEstimators[filterNr]->getEventsInWindow();
    }

private:
    int energyEstimatorCnt;
    MotionEnergyEstimator **motionEnergyEstimators;
    Buffer2D *opponentMotionEnergies;
    long *updateTimeStamps;
    double *filterThresholds;
    QList<float> orientations;
    QList<FilterSettings> settings;
};

#endif // OPTICFLOWESTIMATOR_H
