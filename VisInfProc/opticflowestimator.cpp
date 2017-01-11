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
    opticFlowUpToDate = false;
    motionEnergyEstimators = new MotionEnergyEstimator*[energyEstimatorCnt];
    motionEnergyBuffers = new Buffer2D*[energyEstimatorCnt*orientations.length()];
    updateTimeStamps = new long[energyEstimatorCnt];
    cpuArrGpuMotionEnergies = new float*[energyEstimatorCnt*orientations.length()];
    cudaStreams = new cudaStream_t[energyEstimatorCnt];

    float cpuArrSpeeds[energyEstimatorCnt];
    for(int i = 0; i < energyEstimatorCnt; i++){
        cudaStreamCreate(&cudaStreams[i]);
        motionEnergyEstimators[i] = new MotionEnergyEstimator(settings.at(i),orientations);
        updateTimeStamps[i] = -1;
        cpuArrSpeeds[i] = settings.at(i).speed_px_per_sec;
        for(int j= 0; j < orientations.length();j++){
            int idx = i*orientations.length() + j;
            motionEnergyBuffers[idx] = new Buffer2D(DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_HEIGHT);
            cpuArrGpuMotionEnergies[idx] = motionEnergyBuffers[idx]->getGPUPtr();
        }
    }

    opticFlowVec[0].resize(DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_HEIGHT);
    opticFlowVec[0].fill(0);
    opticFlowVec[1].resize(DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_HEIGHT);
    opticFlowVec[1].fill(0);

    qDebug("Uploading orientations to GPU...");
    gpuArrOrientations = NULL;
    int sz = sizeof(float)*orientations.length();
    gpuErrchk(cudaMalloc(&gpuArrOrientations,sz));
    gpuErrchk(cudaMemcpyAsync(gpuArrOrientations,orientations.data(),sz,cudaMemcpyHostToDevice,cudaStreams[0]));

    qDebug("Uploading pointers array to  motion energy buffers to GPU...");
    gpuArrGpuMotionEnergies = NULL;
    sz = sizeof(float*)*energyEstimatorCnt*orientations.length();
    gpuErrchk(cudaMalloc(&gpuArrGpuMotionEnergies,sz));
    gpuErrchk(cudaMemcpyAsync(gpuArrGpuMotionEnergies,cpuArrGpuMotionEnergies,sz,cudaMemcpyHostToDevice,cudaStreams[0]));

    qDebug("Uploading speeds to GPU...");
    gpuArrSpeeds = NULL;
    sz = sizeof(float)*energyEstimatorCnt;
    gpuErrchk(cudaMalloc(&gpuArrSpeeds,sz));
    gpuErrchk(cudaMemcpyAsync(gpuArrSpeeds,cpuArrSpeeds,sz,cudaMemcpyHostToDevice,cudaStreams[0]));

    cudaStreamSynchronize(cudaStreams[0]);
}

OpticFlowEstimator::~OpticFlowEstimator()
{
    qDebug("Destroying OpticFlow estimator...");
    for(int i = 0; i < energyEstimatorCnt; i++){
        cudaStreamDestroy(cudaStreams[i]);
        delete motionEnergyEstimators[i];
        for(int j= 0; j < orientations.length();j++){
            int idx = i*orientations.length() + j;
            delete motionEnergyBuffers[idx];
        }
    }

    delete[] motionEnergyEstimators;
    motionEnergyEstimators = NULL;
    delete[] updateTimeStamps;
    updateTimeStamps = NULL;

    gpuErrchk(cudaFree(gpuArrOrientations));
    gpuArrOrientations = NULL;
    gpuErrchk(cudaFree(gpuArrGpuMotionEnergies));
    gpuArrGpuMotionEnergies = NULL;
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
    // Start parallel reading of  motion energy
    for(int i = 0; i < energyEstimatorCnt; i++){
        if(somethingToDo[i]){
            updateTimeStamps[i] = motionEnergyEstimators[i]->startReadMotionEnergyAsync(
                        &cpuArrGpuMotionEnergies[i*orientations.length()]);
        }
    }

#ifndef NDEBUG
    nvtxMark("Normalize Result");
#endif
    // Start parallel normalizing of motion energy
    for(int i = 0; i < energyEstimatorCnt; i++){
        if(somethingToDo[i]){
            motionEnergyEstimators[i]->startNormalizeEnergiesAsync(
                        &cpuArrGpuMotionEnergies[i*orientations.length()]);
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
    cudaComputeOpticFlow(opticFlowVec[0].getSizeX(),opticFlowVec[0].getSizeY(),
                opticFlowVec[0].getGPUPtr(),opticFlowVec[1].getGPUPtr(),
                gpuArrGpuMotionEnergies,
                gpuArrOrientations,orientations.length(),
                gpuArrSpeeds,energyEstimatorCnt,
                cudaStreams[0]);

    // Synchronize
    cudaStreamSynchronize(cudaStreams[0]);

    opticFlowUpToDate = true;
#ifndef NDEBUG
    nvtxRangeEnd(id);
#endif
}
