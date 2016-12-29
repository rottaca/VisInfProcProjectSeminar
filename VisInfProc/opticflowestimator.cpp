#include "opticflowestimator.h"

OpticFlowEstimator::OpticFlowEstimator(QList<FilterSettings> settings, QList<float> orientations)
{
    this->orientations = orientations;
    this->settings = settings;
    energyEstimatorCnt = settings.length();
    motionEnergyEstimators = new MotionEnergyEstimator*[energyEstimatorCnt];
    opponentMotionEnergies = new Buffer2D[energyEstimatorCnt*orientations.length()];
    updateTimeStamps = new long[energyEstimatorCnt];
    filterThresholds = new double[energyEstimatorCnt];
    for(int i = 0; i < energyEstimatorCnt; i++){
        motionEnergyEstimators[i] = new MotionEnergyEstimator(settings.at(i),orientations);
        updateTimeStamps[i] = -1;
        filterThresholds[i] = 1;   // Default
    }
    opticFlowVec1.resize(128,128);
    opticFlowVec2.resize(128,128);
}

OpticFlowEstimator::~OpticFlowEstimator()
{
    for(int i = 0; i < energyEstimatorCnt; i++){
        delete motionEnergyEstimators[i];
    }

    delete[] motionEnergyEstimators;
    motionEnergyEstimators = NULL;
    delete[] opponentMotionEnergies;
    opponentMotionEnergies = NULL;
    delete[] updateTimeStamps;
    updateTimeStamps = NULL;
    delete[] filterThresholds;
    filterThresholds = NULL;
}

void OpticFlowEstimator::processEvent(DVSEventHandler::DVSEvent event)
{
    bool updates = false;
    for(int i = 0; i < energyEstimatorCnt; i++){
        motionEnergyEstimators[i]->processEvent(event);
        // New motion energy ready ?
        if(motionEnergyEstimators[i]->isEnergyReady()){
            updates = true;
            for(int j = 0; j < orientations.length(); j++){
                motionEnergyEstimators[i]->getMotionEnergy(j,
                            opponentMotionEnergies[i*orientations.length() + j]);
            }
            updateTimeStamps[i] = event.timestamp;
        }
    }
}
