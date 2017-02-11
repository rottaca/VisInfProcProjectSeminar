#include "opticflowestimator.h"
#include "settings.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_helper.h"
#include <nvToolsExt.h>

#include <QElapsedTimer>

OpticFlowEstimator::OpticFlowEstimator(QVector<FilterSettings> settings, QVector<float> orientations)
{
    this->orientations = orientations;
    this->settings = settings;
    opticFlowUpToDate = false;
    energyEstimatorCnt = settings.length();
    motionEnergyEstimators = new MotionEnergyEstimator*[energyEstimatorCnt];
    motionEnergyBuffers = new Buffer2D*[energyEstimatorCnt*orientations.length()];
    updateTimeStamps = new quint32[energyEstimatorCnt];
    cpuArrGpuMotionEnergies = new float*[energyEstimatorCnt*orientations.length()];
    cudaStreams = new cudaStream_t[energyEstimatorCnt];
    energyThreshold = FLOW_DEFAULT_MIN_ENERGY_THRESHOLD;
    opticFlowSpeed.resize(DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_HEIGHT);
    opticFlowDir.resize(DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_HEIGHT);
    opticFlowEnergy.resize(DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_HEIGHT);

    float cpuArrSpeeds[energyEstimatorCnt];
    for(int i = 0; i < energyEstimatorCnt; i++) {
        cudaStreamCreate(&cudaStreams[i]);
        motionEnergyEstimators[i] = new MotionEnergyEstimator(settings.at(i),orientations);
        cpuArrSpeeds[i] = settings.at(i).speed_px_per_sec;

        for(int j= 0; j < orientations.length(); j++) {
            int idx = i*orientations.length() + j;
            motionEnergyBuffers[idx] = new Buffer2D(DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_HEIGHT);
            cpuArrGpuMotionEnergies[idx] = motionEnergyBuffers[idx]->getGPUPtr();
        }
    }
    reset();

    PRINT_DEBUG("Uploading speeds and orientations to GPU...");
    gpuArrOrientations = NULL;
    size_t sz = sizeof(float)*orientations.length();
    gpuErrchk(cudaMalloc(&gpuArrOrientations,sz));
    gpuErrchk(cudaMemcpyAsync(gpuArrOrientations,orientations.data(),sz,cudaMemcpyHostToDevice,cudaStreams[0]));

    gpuArrGpuMotionEnergies = NULL;
    sz = sizeof(float*)*energyEstimatorCnt*orientations.length();
    gpuErrchk(cudaMalloc(&gpuArrGpuMotionEnergies,sz));
    gpuErrchk(cudaMemcpyAsync(gpuArrGpuMotionEnergies,cpuArrGpuMotionEnergies,sz,cudaMemcpyHostToDevice,cudaStreams[0]));

    gpuArrSpeeds = NULL;
    sz = sizeof(float)*energyEstimatorCnt;
    gpuErrchk(cudaMalloc(&gpuArrSpeeds,sz));
    gpuErrchk(cudaMemcpyAsync(gpuArrSpeeds,cpuArrSpeeds,sz,cudaMemcpyHostToDevice,cudaStreams[0]));

    cudaStreamSynchronize(cudaStreams[0]);
}

OpticFlowEstimator::~OpticFlowEstimator()
{
    PRINT_DEBUG("Destroying OpticFlow estimator...");
    for(int i = 0; i < energyEstimatorCnt; i++) {
        cudaStreamDestroy(cudaStreams[i]);
        delete motionEnergyEstimators[i];
        for(int j= 0; j < orientations.length(); j++) {
            int idx = i*orientations.length() + j;
            delete motionEnergyBuffers[idx];
        }
    }

    delete[] motionEnergyEstimators;
    motionEnergyEstimators = NULL;
    delete[] updateTimeStamps;
    updateTimeStamps = NULL;
    delete[] motionEnergyBuffers;
    motionEnergyBuffers = NULL;
    delete[] cpuArrGpuMotionEnergies;
    cpuArrGpuMotionEnergies = NULL;
    delete[] cudaStreams;
    cudaStreams = NULL;

    gpuErrchk(cudaFree(gpuArrOrientations));
    gpuArrOrientations = NULL;
    gpuErrchk(cudaFree(gpuArrGpuMotionEnergies));
    gpuArrGpuMotionEnergies = NULL;
}

void OpticFlowEstimator::reset()
{
    opticFlowUpToDate = false;
    opticFlowSpeed.fill(0);
    opticFlowDir.fill(0);
    opticFlowEnergy.fill(0);

    motionEnergyMutex.lock();
    for(int i = 0; i < energyEstimatorCnt; i++) {
        updateTimeStamps[i] = UINT32_MAX;
        motionEnergyEstimators[i]->reset();
    }
    motionEnergyMutex.unlock();
}

bool OpticFlowEstimator::onNewEvent(const DVSEvent& e)
{
    bool dataRead = false;
    for(int i = 0; i < energyEstimatorCnt; i++) {
        dataRead |= motionEnergyEstimators[i]->onNewEvent(e);
    }
    return dataRead;
}

void OpticFlowEstimator::process()
{
    // Check where we have something to do
    bool somethingToDo[energyEstimatorCnt];
    int cnt = 0;
    for(int i = 0; i < energyEstimatorCnt; i++) {
        somethingToDo[i] = motionEnergyEstimators[i]->isEventListReady();
        if(somethingToDo[i])
            cnt++;
    }
    // Nothing to do
    if(cnt == 0)
        return;
    //QElapsedTimer timer;
    //size_t eventCnt = motionEnergyEstimators[0]->getEventsInCurrSlot();
    //timer.start();
#ifdef DEBUG_INSERT_PROFILER_MARKS
    nvtxRangeId_t id = nvtxRangeStart("Processing Block");
    nvtxMark("Upload Events");
#endif
    // Upload event lists to gpu
    for(int i = 0; i < energyEstimatorCnt; i++) {
        if(somethingToDo[i])
            motionEnergyEstimators[i]->uploadEvents();
    }

#ifdef DEBUG_INSERT_PROFILER_MARKS
    nvtxMark("Process events");
#endif
    //qDebug("Start processing");
    // Start parallel batch processing of events
    for(int i = 0; i < energyEstimatorCnt; i++) {
        if(somethingToDo[i])
            motionEnergyEstimators[i]->startProcessEventsBatchAsync();
    }

#ifdef DEBUG_INSERT_PROFILER_MARKS
    nvtxMark("Read Result");
#endif
    // qDebug("Read result");
    motionEnergyMutex.lock();
    // Start parallel reading of  motion energy
    for(int i = 0; i < energyEstimatorCnt; i++) {
        if(somethingToDo[i]) {
            updateTimeStamps[i] = motionEnergyEstimators[i]->startReadMotionEnergyAsync(
                                      &cpuArrGpuMotionEnergies[i*orientations.length()]);
        }
    }

#ifdef DEBUG_INSERT_PROFILER_MARKS
    nvtxMark("Normalize Result");
#endif
    // Start parallel normalizing of motion energy
    for(int i = 0; i < energyEstimatorCnt; i++) {
        if(somethingToDo[i]) {
            motionEnergyEstimators[i]->startNormalizeEnergiesAsync(
                &cpuArrGpuMotionEnergies[i*orientations.length()]);
        }
    }
#ifdef DEBUG_INSERT_PROFILER_MARKS
    nvtxMark("Sync streams");
#endif
    //qDebug("Sync");
    // Syncronize all streams and wait for them to finish
    for(int i = 0; i < energyEstimatorCnt; i++) {
        if(somethingToDo[i]) {
            motionEnergyEstimators[i]->syncStreams();
        }
    }
    // The optic flow stored in member variables is old
    opticFlowUpToDate = false;
    motionEnergyMutex.unlock();
    //qDebug("%lu %llu",eventCnt,timer.nsecsElapsed()/1000);

#ifdef DEBUG_INSERT_PROFILER_MARKS
    nvtxRangeEnd(id);
#endif
}

void OpticFlowEstimator::computeOpticFlow()
{
#ifdef DEBUG_INSERT_PROFILER_MARKS
    nvtxRangeId_t id = nvtxRangeStart("Optic Flow");
#endif

    cudaComputeFlow(DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_HEIGHT,
                    opticFlowEnergy.getGPUPtr(),opticFlowDir.getGPUPtr(),opticFlowSpeed.getGPUPtr(),
                    gpuArrGpuMotionEnergies,
                    gpuArrOrientations,orientations.length(),
                    gpuArrSpeeds,energyEstimatorCnt,
                    energyThreshold,
                    cudaStreams[0]);
    cudaStreamSynchronize(cudaStreams[0]);
    opticFlowUpToDate = true;

#ifdef DEBUG_INSERT_PROFILER_MARKS
    nvtxRangeEnd(id);
#endif
}
