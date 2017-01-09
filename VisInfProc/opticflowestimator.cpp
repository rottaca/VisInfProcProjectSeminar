#include "opticflowestimator.h"
#include "settings.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_helper.h"
#include <nvToolsExt.h>

OpticFlowEstimator::OpticFlowEstimator(QVector<FilterSettings> settings, QVector<float> orientations)
{
    this->orientations = orientations;
    this->settings = settings;
    energyEstimatorCnt = settings.length();
    motionEnergyEstimators = new MotionEnergyEstimator*[energyEstimatorCnt];
    opponentMotionEnergies = new Buffer2D*[energyEstimatorCnt*orientations.length()];
    updateTimeStamps = new long[energyEstimatorCnt];

    gpuOpponentMotionEnergies = new float*[energyEstimatorCnt*orientations.length()];
    cudaStreams = new cudaStream_t[energyEstimatorCnt];
    for(int i = 0; i < energyEstimatorCnt; i++){
        cudaStreamCreate(&cudaStreams[i]);
        motionEnergyEstimators[i] = new MotionEnergyEstimator(settings.at(i),orientations);
        updateTimeStamps[i] = -1;
        for(int j= 0; j < orientations.length();j++){
            int idx = i*orientations.length() + j;
            opponentMotionEnergies[idx] = new Buffer2D(DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_HEIGHT);
            gpuOpponentMotionEnergies[idx] = opponentMotionEnergies[idx]->getGPUPtr();
        }
    }

    opticFlowVec[0].resize(DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_HEIGHT);
    opticFlowVec[0].fill(0);
    opticFlowVec[1].resize(DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_HEIGHT);
    opticFlowVec[1].fill(0);

    gpuOrientations = NULL;
    int sz = sizeof(float)*orientations.length();
    gpuErrchk(cudaMalloc(&gpuOrientations,sz));
    gpuErrchk(cudaMemcpyAsync(gpuOrientations,orientations.data(),sz,cudaMemcpyHostToDevice,cudaStreams[0]));
    gpuArrgpuOpponentMotionEnergies = NULL;
    sz = sizeof(float*)*energyEstimatorCnt*orientations.length();
    gpuErrchk(cudaMalloc(&gpuArrgpuOpponentMotionEnergies,sz));
    gpuErrchk(cudaMemcpyAsync(gpuArrgpuOpponentMotionEnergies,gpuOpponentMotionEnergies,sz,cudaMemcpyHostToDevice,cudaStreams[0]));

    cudaStreamSynchronize(cudaStreams[0]);
}

OpticFlowEstimator::~OpticFlowEstimator()
{

    for(int i = 0; i < energyEstimatorCnt; i++){
        cudaStreamDestroy(cudaStreams[i]);
        delete motionEnergyEstimators[i];
        for(int j= 0; j < orientations.length();j++){
            int idx = i*orientations.length() + j;
            delete opponentMotionEnergies[idx];
        }
    }

    delete[] motionEnergyEstimators;
    motionEnergyEstimators = NULL;
    delete[] updateTimeStamps;
    updateTimeStamps = NULL;

    gpuErrchk(cudaFree(gpuOrientations));
    gpuOrientations = NULL;
    gpuErrchk(cudaFree(gpuArrgpuOpponentMotionEnergies));
    gpuArrgpuOpponentMotionEnergies = NULL;
}

void OpticFlowEstimator::onNewEvent(const SerialeDVSInterface::DVSEvent& e)
{
    for(int i = 0; i < energyEstimatorCnt; i++){
        motionEnergyEstimators[i]->onNewEvent(e);
    }
}

void OpticFlowEstimator::process()
{
    // Check where we have something to do
    bool somethingToDo[energyEstimatorCnt];
    int cnt = 0;
    for(int i = 0; i < energyEstimatorCnt; i++){
        somethingToDo[i] = motionEnergyEstimators[i]->isEventListReady();
        if(somethingToDo[i])
            cnt++;
    }
    // Nothing to do
    if(cnt == 0)
        return;

#ifndef NDEBUG
    nvtxRangeId_t id = nvtxRangeStart("Processing Block");
    nvtxMark("Upload Events");
#endif
    // Upload event lists to gpu
    //qDebug("Upload events");
    for(int i = 0; i < energyEstimatorCnt; i++){
        if(somethingToDo[i])
            motionEnergyEstimators[i]->startUploadEventsAsync();
    }

#ifndef NDEBUG
    nvtxMark("Process events");
#endif
    //qDebug("Start processing");
    // Start parallel batch processing of events
    for(int i = 0; i < energyEstimatorCnt; i++){
        if(somethingToDo[i])
            motionEnergyEstimators[i]->startProcessEventsBatchAsync();
    }

#ifndef NDEBUG
    nvtxMark("Read Result");
#endif
   // qDebug("Read result");
    motionEnergyMutex.lock();
    // Start parallel reading of opponent motion energy
    for(int i = 0; i < energyEstimatorCnt; i++){
        if(somethingToDo[i]){
            updateTimeStamps[i] = motionEnergyEstimators[i]->startReadMotionEnergyAsync(
                        &gpuOpponentMotionEnergies[i*orientations.length()]);
        }
    }
#ifndef NDEBUG
    nvtxMark("Sync streams");
#endif
    //qDebug("Sync");
    // Syncronize all streams and wait for them to finish
    for(int i = 0; i < energyEstimatorCnt; i++){
        if(somethingToDo[i]){
            motionEnergyEstimators[i]->syncStreams();
        }
    }
    // The optic flow stored in member variables is old
    opticFlowUpToDate = false;
    motionEnergyMutex.unlock();

#ifndef NDEBUG
    nvtxRangeEnd(id);
#endif
}

void OpticFlowEstimator::computeOpticFlow(){

#ifndef NDEBUG
    nvtxRangeId_t id = nvtxRangeStart("Optic Flow");
#endif
    // Start optic flow computation
    for(int i = 0; i < 1; i++){
        cudaComputeOpticFlow(opticFlowVec[0].getSizeX(),opticFlowVec[0].getSizeY(),
                opticFlowVec[0].getGPUPtr(),opticFlowVec[1].getGPUPtr(),
                gpuArrgpuOpponentMotionEnergies + orientations.length()*i,gpuOrientations,orientations.length(),
                cudaStreams[i]);
    }
    // Synchronize
    for(int i = 0; i < energyEstimatorCnt; i++){
        cudaStreamSynchronize(cudaStreams[i]);
    }
    opticFlowUpToDate = true;
#ifndef NDEBUG
    nvtxRangeEnd(id);
#endif
}
