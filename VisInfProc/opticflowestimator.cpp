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
    opticFlowEnergyUpToDate = new bool[energyEstimatorCnt];
    motionEnergyEstimators = new MotionEnergyEstimator*[energyEstimatorCnt];
    motionEnergyBuffers = new Buffer2D*[energyEstimatorCnt*orientations.length()];
    updateTimeStamps = new quint32[energyEstimatorCnt];
    cpuArrGpuMotionEnergies = new float*[energyEstimatorCnt*orientations.length()];
    cudaStreams = new cudaStream_t[energyEstimatorCnt];
    opticFlowEnergies = new Buffer2D[energyEstimatorCnt];
    opticFlowDirs = new Buffer2D[energyEstimatorCnt];
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

        opticFlowEnergies[i].resize(DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_HEIGHT);
        opticFlowDirs[i].resize(DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_HEIGHT);
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
    delete[] opticFlowEnergyUpToDate;
    opticFlowEnergyUpToDate = NULL;
    delete[] opticFlowEnergyUpToDate;
    opticFlowEnergyUpToDate = NULL;
    delete[] motionEnergyBuffers;
    motionEnergyBuffers = NULL;
    delete[] cpuArrGpuMotionEnergies;
    cpuArrGpuMotionEnergies = NULL;
    delete[] cudaStreams;
    cudaStreams = NULL;
    delete[] opticFlowEnergies;
    opticFlowEnergies = NULL;
    delete[] opticFlowDirs;
    opticFlowDirs = NULL;

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
        opticFlowEnergyUpToDate[i] = false;
        updateTimeStamps[i] = UINT32_MAX;
        opticFlowEnergies[i].fill(0);
        opticFlowDirs[i].fill(0);
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
#ifndef NDEBUG
    nvtxRangeId_t id = nvtxRangeStart("Processing Block");
    nvtxMark("Upload Events");
#endif
    // Upload event lists to gpu
    for(int i = 0; i < energyEstimatorCnt; i++) {
        if(somethingToDo[i])
            motionEnergyEstimators[i]->uploadEvents();
    }

#ifndef NDEBUG
    nvtxMark("Process events");
#endif
    //qDebug("Start processing");
    // Start parallel batch processing of events
    for(int i = 0; i < energyEstimatorCnt; i++) {
        if(somethingToDo[i])
            motionEnergyEstimators[i]->startProcessEventsBatchAsync();
    }

#ifndef NDEBUG
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

#ifndef NDEBUG
    nvtxMark("Normalize Result");
#endif
    // Start parallel normalizing of motion energy
    for(int i = 0; i < energyEstimatorCnt; i++) {
        if(somethingToDo[i]) {
            motionEnergyEstimators[i]->startNormalizeEnergiesAsync(
                &cpuArrGpuMotionEnergies[i*orientations.length()]);
        }
    }
#ifndef NDEBUG
    nvtxMark("Sync streams");
#endif
    //qDebug("Sync");
    // Syncronize all streams and wait for them to finish
    for(int i = 0; i < energyEstimatorCnt; i++) {
        if(somethingToDo[i]) {
            motionEnergyEstimators[i]->syncStreams();
            // The optic flow stored in member variables is old
            opticFlowEnergyUpToDate[i] = false;
            opticFlowUpToDate = false;
        }
    }
    motionEnergyMutex.unlock();
    //qDebug("%lu %llu",eventCnt,timer.nsecsElapsed()/1000);

#ifndef NDEBUG
    nvtxRangeEnd(id);
#endif
}

void OpticFlowEstimator::computeOpticFlowEnergy(int speedIdx)
{

#ifndef NDEBUG
    nvtxRangeId_t id = nvtxRangeStart("Optic Flow Energy");
#endif

    // Start optic flow computation
    cudaComputeFlowEnergyAndDir(opticFlowEnergies[speedIdx].getSizeX(),opticFlowEnergies[speedIdx].getSizeY(),
                                opticFlowEnergies[speedIdx].getGPUPtr(),opticFlowDirs[speedIdx].getGPUPtr(),
                                gpuArrGpuMotionEnergies + orientations.length()*speedIdx,
                                gpuArrOrientations,orientations.length(),
                                settings[speedIdx].speed_px_per_sec,
                                cudaStreams[0]);

    // Synchronize
    cudaStreamSynchronize(cudaStreams[0]);

    opticFlowEnergyUpToDate[speedIdx] = true;
#ifndef NDEBUG
    nvtxRangeEnd(id);
#endif
}

void OpticFlowEstimator::computeOpticFlow()
{
#ifndef NDEBUG
    nvtxRangeId_t id = nvtxRangeStart("Optic Flow");
#endif

    // Get all optic flows
    int speeds = settings.length();
    float* opticFlowEnergyPtr[speeds];
    float* opticFlowDirPtr[speeds];

    for(int i = 0; i < speeds; i++) {
        opticFlowEnergyPtr[i] = opticFlowEnergies[i].getCPUPtr();
        opticFlowDirPtr[i] = opticFlowDirs[i].getCPUPtr();
    }
    int sx = DVS_RESOLUTION_WIDTH;
    int sy = DVS_RESOLUTION_HEIGHT;

    float* combinedFlowSpeedPtr = opticFlowSpeed.getCPUPtr();
    float* combinedFlowDirPtr = opticFlowDir.getCPUPtr();
    float* combinedFlowEnergyPtr = opticFlowEnergy.getCPUPtr();

    for(int i = 0; i < sx*sy; i++) {
        float outSpeed = 0;
        float outDir = 0;
        float outEnergy = 0;
        int maxIdx = -1;
        for(int j = 0; j < speeds; j++) {
            float energy = opticFlowEnergyPtr[j][i];
            float dir = opticFlowDirPtr[j][i];

            if(energy >= energyThreshold && energy > outEnergy) {
                outSpeed = settings.at(j).speed_px_per_sec;
                outDir = dir;
                outEnergy = energy;
                maxIdx = j;
            }
        }
#ifndef DISABLE_INTERPOLATION
        if(maxIdx >= 0) {
            float x1,x2,x3,y1,y2,y3;
            int idxCenter;
            // Find x and y coordinates for interpolation
            if(maxIdx == 0) {
                idxCenter = maxIdx+1;
            } else if(maxIdx == speeds-1) {
                idxCenter = maxIdx-1;
            } else {
                idxCenter = maxIdx;
            }

            x1 = settings.at(idxCenter-1).speed_px_per_sec;
            x2 = settings.at(idxCenter).speed_px_per_sec;
            x3 = settings.at(idxCenter+1).speed_px_per_sec;

            y1 = opticFlowEnergyPtr[idxCenter-1][i];
            y2 = opticFlowEnergyPtr[idxCenter][i];
            y3 = opticFlowEnergyPtr[idxCenter+1][i];

            float d2 = 2*((y3-y2)/(x3-x2)-(y1-y2)/(x1-x2))/(x3-x1);
            float d1 = 0;
            if ((x3+x1)>=(x2+x2))
                d1 = (y3-y2)/(x3-x2) - 0.5*d2*(x3-x2);
            else
                d1 = (y2-y1)/(x2-x1) + 0.5*d2*(x2-x1);
            if(d2 < 0) {
                float xe = x2 - d1/d2;
                if(xe >= x1 && xe <= x3) {
                    float ye = y2 + 0.5*d1*(xe-x2);
                    // ye is interpolated flow energy for speed xe
                    // Next step: Decompose in x and y direction
                    // Interpolate orientation linear between the left and right point from xe
                    if(xe < x2) {
                        // Interpolation parameter for (x1,y1)
                        float t = (xe - x1)/(x2-x1);

                        float dir = opticFlowDirPtr[idxCenter-1][i];
                        float dir2 = opticFlowDirPtr[idxCenter][i];
                        // Scale to new speed
                        outSpeed = xe;
                        outEnergy = ye;
                        outDir = dir*t+dir2*(1-t);
                        //outX = 50*t;
                        //outY = 50*(1-t);
                    } else {
                        // Interpolation parameter for (x3,y3)
                        float t = (x3 - xe)/(x3-x2);

                        float dir = opticFlowDirPtr[idxCenter+1][i];
                        float dir2 = opticFlowDirPtr[idxCenter][i];
                        // Scale to new speed
                        outSpeed = xe;
                        outEnergy = ye;
                        outDir = dir*t+dir2*(1-t);
                        //outX = 50*t;
                        //outY = 50*(1-t);
                    }
#ifdef DEBUG_FLOW_DIR_ENCODE_INTERPOLATION
                    // Interpolated speed and orientation: red
                    outDir = 0.f/180*M_PI;
                    outSpeed = 60;
#endif
                }
#ifdef DEBUG_FLOW_DIR_ENCODE_INTERPOLATION
                else {
                    // Maxima outside range
                    if(xe < x1) {
                        outDir = 90.f/180*M_PI; // Slower: purple
                    } else {
                        outDir = 180.f/180*M_PI;    // Faster: bright blue
                    }
                    outSpeed = 60;
                }
#endif
            } else {
                // No clear answer possible
                outSpeed = 0;
                outDir = 0;
#ifdef DEBUG_FLOW_DIR_ENCODE_INTERPOLATION
                // Parabel has no maximum, use strongest response: green
                outDir = 270.f/180*M_PI;
                outSpeed = 60;
#endif
            }
        }
#endif
        combinedFlowSpeedPtr[i] = outSpeed;
        combinedFlowDirPtr[i] = outDir;
        combinedFlowEnergyPtr[i] = outEnergy;
    }

    opticFlowUpToDate = true;
#ifndef NDEBUG
    nvtxRangeEnd(id);
#endif
}
