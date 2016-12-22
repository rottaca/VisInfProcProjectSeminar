#include "opticflowestimator.h"

OpticFlowEstimator::OpticFlowEstimator(QList<FilterSettings> settings, QList<float> orientations)
{
    this->orientations = orientations;
    this->settings = settings;
    motionEnergyEstimators = new MotionEnergyEstimator*[settings.length()];
    for(int i = 0; i < settings.length(); i++){
        motionEnergyEstimators[i] = new MotionEnergyEstimator(settings.at(i),orientations);
    }
}

OpticFlowEstimator::~OpticFlowEstimator()
{
    if(motionEnergyEstimators != NULL){
        for(int i = 0; i < settings.length(); i++){
            delete motionEnergyEstimators[i];
        }

        delete[] motionEnergyEstimators;
    }

    motionEnergyEstimators = NULL;
}
