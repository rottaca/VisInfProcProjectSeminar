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

    void onNewEvent(const DVSEventHandler::DVSEvent &e);
    void process();

    long getMotionEnergy(int filterNr, int orientationIdx, Buffer2D &opponentMotionEnergy){
        assert(filterNr >= 0);
        assert(filterNr < energyEstimatorCnt);
        assert(orientationIdx >= 0);
        assert(orientationIdx < orientations.length());
        motionEnergyMutex.lock();
        opponentMotionEnergy = *(opponentMotionEnergies[filterNr*orientations.length() + orientationIdx]);
        motionEnergyMutex.unlock();
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

    QList<double> orientations;
    QList<FilterSettings> settings;

    Buffer2D opticFlowVec[2];

    QMutex motionEnergyMutex;
    Buffer2D **opponentMotionEnergies;
    double **gpuOpponentMotionEnergies;
    long *updateTimeStamps;
};

#endif // OPTICFLOWESTIMATOR_H
