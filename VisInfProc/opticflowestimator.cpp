#include "opticflowestimator.h"

OpticFlowEstimator::OpticFlowEstimator(QList<FilterSettings> settings, QList<float> orientations)
{
    this->orientations = orientations;
    this->settings = settings;
    energyEstimatorCnt = settings.length();
    motionEnergyEstimators = new MotionEnergyEstimator*[energyEstimatorCnt];
    motionEnergiesLeft = new Buffer2D[energyEstimatorCnt*orientations.length()];
    motionEnergiesRight = new Buffer2D[energyEstimatorCnt*orientations.length()];
    updateTimeStamps = new long[energyEstimatorCnt];
    filterThresholds = new double[energyEstimatorCnt];
    for(int i = 0; i < energyEstimatorCnt; i++){
        motionEnergyEstimators[i] = new MotionEnergyEstimator(settings.at(i),orientations);
        updateTimeStamps[i] = -1;
        filterThresholds[i] = 1;   // Default
    }
}

OpticFlowEstimator::~OpticFlowEstimator()
{
    for(int i = 0; i < energyEstimatorCnt; i++){
        delete motionEnergyEstimators[i];
    }

    delete[] motionEnergyEstimators;
    motionEnergyEstimators = NULL;
    delete motionEnergiesLeft;
    motionEnergiesLeft = NULL;
    delete motionEnergiesRight;
    motionEnergiesRight = NULL;
    delete updateTimeStamps;
    updateTimeStamps = NULL;
    delete filterThresholds;
    filterThresholds = NULL;
}

void OpticFlowEstimator::processEvent(DVSEventHandler::DVSEvent event)
{
    for(int i = 0; i < energyEstimatorCnt; i++){
        motionEnergyEstimators[i]->processEvent(event);
        // Rew motion energy ready ?
        if(motionEnergyEstimators[i]->isEnergyReady()){
            for(int j = 0; j < orientations.length(); j++){
                motionEnergyEstimators[i]->getMotionEnergy(j,
                            motionEnergiesLeft[i*orientations.length() + j],
                        motionEnergiesRight[i*orientations.length() + j]);
            }
            updateTimeStamps[i] = event.timestamp;
        }
    }
}
