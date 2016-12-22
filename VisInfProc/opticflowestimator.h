#ifndef OPTICFLOWESTIMATOR_H
#define OPTICFLOWESTIMATOR_H

#include "motionenergyestimator.h"

#include <QList>

class OpticFlowEstimator
{
public:
    OpticFlowEstimator(QList<FilterSettings> settings, QList<float> orientations);
    ~OpticFlowEstimator();

private:
    MotionEnergyEstimator **motionEnergyEstimators;
    QList<float> orientations;
    QList<FilterSettings> settings;
};

#endif // OPTICFLOWESTIMATOR_H
