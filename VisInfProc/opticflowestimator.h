#ifndef OPTICFLOWESTIMATOR_H
#define OPTICFLOWESTIMATOR_H

#include "motionenergyestimator.h"
#include "dvseventhandler.h"

#include <QList>
#include <assert.h>
#include <QVector>

extern void cudaComputeOpticFlow(int sx, int sy,
                                 double* gpuFlowX,double* gpuFlowY,
                                 double** gpuEnergy,double* orientations, int orientationCnt);

class OpticFlowEstimator
{
public:
    OpticFlowEstimator(QList<FilterSettings> settings, QList<double> orientations);
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

    void getOpticFlow(Buffer2D &flowX, Buffer2D &flowY){
        flowX = opticFlowVec[0];
        flowY = opticFlowVec[1];
    }


    QVector<DVSEventHandler::DVSEvent> getEventsInWindow(int filterNr){
        assert(filterNr >= 0);
        assert(filterNr < energyEstimatorCnt);
        return motionEnergyEstimators[filterNr]->getEventsInWindow();
    }

private:
    void computeOpticFlow();

private:
    int energyEstimatorCnt;
    MotionEnergyEstimator **motionEnergyEstimators;
    Buffer2D *opponentMotionEnergies;
    long *updateTimeStamps;
    double *filterThresholds;
    QList<double> orientations;
    QList<FilterSettings> settings;

    Buffer2D opticFlowVec[2];
};

#endif // OPTICFLOWESTIMATOR_H
