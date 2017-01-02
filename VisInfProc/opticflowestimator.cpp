#include "opticflowestimator.h"
#include "settings.h"

OpticFlowEstimator::OpticFlowEstimator(QList<FilterSettings> settings, QList<double> orientations)
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
    opticFlowVec[0].resize(DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_HEIGHT);
    opticFlowVec[0].fill(0);
    opticFlowVec[1].resize(DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_HEIGHT);
    opticFlowVec[1].fill(0);
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
            updateTimeStamps[i] = event.timestamp;
            for(int j = 0; j < orientations.length(); j++){
                motionEnergyEstimators[i]->getMotionEnergy(j,
                            opponentMotionEnergies[i*orientations.length() + j]);
            }
        }
    }
    // Recompute optic flow
    if(updates){
        computeOpticFlow();
    }
}

void OpticFlowEstimator::computeOpticFlow(){
    // TODO Extend for more filtersettings
    double *energies[orientations.length()];
    double orientationArr[orientations.length()];
    for(int i = 0; i < orientations.length();i++){
        energies[i] = opponentMotionEnergies[i].getGPUPtr();
        orientationArr[i] = orientations.at(i);
    }

    cudaComputeOpticFlow(opticFlowVec[0].getSizeX(),opticFlowVec[0].getSizeY(),
            opticFlowVec[0].getGPUPtr(),opticFlowVec[1].getGPUPtr(),
            energies,orientationArr,orientations.length());
}
