#include "motionenergyestimator.h"
#include <QtMath>
#include <QElapsedTimer>
#include <QPainter>
#include <assert.h>
#include <iostream>
#include <QFile>
#include "settings.h"
#include <assert.h>
#include <nvToolsExt.h>

MotionEnergyEstimator::MotionEnergyEstimator(FilterSettings fs, QVector<float> orientations)
{
    assert(orientations.length() > 0);

    this->fsettings = fs;
    this->orientations = orientations;

    eventsR = &timeSlotEvents[0];
    eventsW = &timeSlotEvents[1];
    eventsR->currWindowStartTime = -1;
    eventsR->events.clear();
    eventsR->slotsToSkip = 0;
    eventsW->currWindowStartTime = -1;
    eventsW->events.clear();
    eventsW->slotsToSkip = 0;

    startTime = -1;
    timePerSlot =(float)fs.timewindow_us/fs.temporalSteps;
    eventListReady = false;
    ringBufferIdx = 0;
    bufferFilterCount = orientations.length()*FILTERS_PER_ORIENTATION;
    fset = new FilterSet*[orientations.length()];
    cpuArrCpuConvBuffers = new Buffer3D*[bufferFilterCount];

    cpuArrGpuFilters = new float* [bufferFilterCount];
    cpuArrGpuConvBuffers = new float* [bufferFilterCount];
    cudaStreams = new cudaStream_t[bufferFilterCount];

    for(int i = 0; i < orientations.length(); i++){
        fset[i] = new FilterSet(fs,orientations.at(i));
        for(int j = 0; j < FILTERS_PER_ORIENTATION; j++)
        {
            cudaStreamCreate(&cudaStreams[i*FILTERS_PER_ORIENTATION+j]);
            cpuArrCpuConvBuffers[i*FILTERS_PER_ORIENTATION+j] = new Buffer3D(DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_HEIGHT,fset[i]->sz);
            cpuArrGpuConvBuffers[i*FILTERS_PER_ORIENTATION+j] = cpuArrCpuConvBuffers[i*FILTERS_PER_ORIENTATION+j]->getGPUPtr();
        }
        cpuArrGpuFilters[i*FILTERS_PER_ORIENTATION    ] = fset[i]->spatialTemporal[FilterSet::PHASE1].getGPUPtr();
        cpuArrGpuFilters[i*FILTERS_PER_ORIENTATION + 1] = fset[i]->spatialTemporal[FilterSet::PHASE2].getGPUPtr();
    }
    // Extract buffer sizes
    fsx = fset[0]->spatialTemporal[FilterSet::PHASE1].getSizeX();
    fsy = fset[0]->spatialTemporal[FilterSet::PHASE1].getSizeY();
    fsz = fset[0]->spatialTemporal[FilterSet::PHASE1].getSizeZ();
    bsx = DVS_RESOLUTION_WIDTH;
    bsy = DVS_RESOLUTION_HEIGHT;
    bsz = fsz;

    gpuEventList = NULL;
    gpuEventListSize = 0;
    eventsAll = 0;
    eventsSkipped = 0;
    gpuEventListSizeAllocated = 0;
    lastEventTime = 0;
}

MotionEnergyEstimator::~MotionEnergyEstimator()
{
    qDebug("Destroying motion energy estimator...");
    qDebug("Peak event count per computation: %d",gpuEventListSizeAllocated);
    for(int i = 0; i < orientations.length(); i++){
        delete fset[i];
        for(int j = 0; j < FILTERS_PER_ORIENTATION; j++){
            gpuErrchk(cudaStreamDestroy(cudaStreams[i*FILTERS_PER_ORIENTATION+j]));
            delete cpuArrCpuConvBuffers[i*FILTERS_PER_ORIENTATION+j];
        }
    }

    delete[] cudaStreams;
    cudaStreams = NULL;
    delete[] fset;
    fset = NULL;
    delete[] cpuArrCpuConvBuffers;
    cpuArrCpuConvBuffers = NULL;
    delete[] cpuArrGpuFilters;
    cpuArrGpuFilters = NULL;
    delete[] cpuArrGpuConvBuffers;
    cpuArrGpuConvBuffers = NULL;

    if(gpuEventList != NULL)
        gpuErrchk(cudaFree(gpuEventList));
}
bool MotionEnergyEstimator::onNewEvent(const eDVSInterface::DVSEvent &e){

    eventStatisticsMutex.lock();
    eventsAll++;
    eventStatisticsMutex.unlock();
    QMutexLocker locker(&eventWriteMutex);

    SimpleEvent ev;
    ev.x = e.posX;
    ev.y = e.posY;

    // Get time from first event as reference
    if(startTime == -1){
        startTime = e.timestamp;
        lastEventTime = startTime;
        eventsW->currWindowStartTime = startTime;
    }
    // Do we have an event with older timestamp?
    else if (lastEventTime > e.timestamp){
        qDebug("Event is not in order according to timestamp! Skip.");
        return false;
    }

    lastEventTime = startTime;

    int deltaT = e.timestamp - eventsW->currWindowStartTime;
    // Do we have to skip any timeslots ? Is the new event too new for the current slot ?
    int timeSlotsToSkip = qFloor((float)deltaT/timePerSlot);

    //qDebug("Skip Slots: %d",timeSlotsToSkip);
    if(timeSlotsToSkip != 0){
        // Flip lists
        eventReadMutex.lock();
        // Was the last block not processed by the worker thread ? Then it is lost
        if(eventListReady && eventsR->events.length() > 0){
            //qDebug("Events skipped: %d",eventsR->events.length());
            eventStatisticsMutex.lock();
            eventsSkipped+=eventsR->events.length();
            eventStatisticsMutex.unlock();
        }
        //qDebug("Flip lists");
        SlotEventData* eventsROld = eventsR;
        eventsR = eventsW;
        eventsW = eventsROld;

        eventsR->slotsToSkip=eventsROld->slotsToSkip+timeSlotsToSkip;
        eventsW->slotsToSkip = 0;
        eventsW->currWindowStartTime=eventsR->currWindowStartTime + timePerSlot*timeSlotsToSkip;

        if(eventsR->events.length() > 0)
            eventListReady = true;
        //qDebug("Lists flipped");
        eventReadMutex.unlock();
        // Clear events for new data
        eventsW->events.clear();
    }
    eventsW->events.append(ev);
    //qDebug("Event added");

    // Update events in timewindow
    eventsInWindowMutex.lock();
    timeWindowEvents.push_back(e);
    while(timeWindowEvents.size() > 0
       && timeWindowEvents.front().timestamp < eventsW->currWindowStartTime - fsettings.timewindow_us)
                    timeWindowEvents.pop_front();
    eventsInWindowMutex.unlock();

    return eventListReady;
}

void MotionEnergyEstimator::startUploadEventsAsync()
{
    eventReadMutex.lock();
    if(eventsR->events.size() > 0)
    {
        int cnt = eventsR->events.length();

        // Do we need more memory for the next event list ?
        if(gpuEventListSizeAllocated < cnt){
            // Delete old list if it exists
            if(gpuEventList != NULL){
#ifndef NDEBUG
                nvtxMark("Release prev Event list");
#endif
                gpuErrchk(cudaFree(gpuEventList));
            }

#ifndef NDEBUG
            nvtxMark("Alloc new Event list");
#endif
            // Allocate buffer for event list
            gpuErrchk(cudaMalloc(&gpuEventList,sizeof(SimpleEvent)*cnt));
            gpuEventListSizeAllocated = cnt;
        }

        // Start asynchonous memcpy
#ifndef NDEBUG
        nvtxMark("Copy events");
#endif
        gpuErrchk(cudaMemcpyAsync(gpuEventList,eventsR->events.data(),
                                  sizeof(SimpleEvent)*cnt,cudaMemcpyHostToDevice,
                                  cudaStreams[DEFAULT_STREAM_ID]));
        gpuEventListSize = cnt;

        // Clear list and wait for next one
        eventsR->events.clear();
    }
    eventReadMutex.unlock();

    eventWriteMutex.lock();
    eventListReady = false;
    eventWriteMutex.unlock();
}

void MotionEnergyEstimator::startProcessEventsBatchAsync()
{
    assert(gpuEventList != NULL);
    cudaStreamSynchronize(cudaStreams[DEFAULT_STREAM_ID]);

    for(int i = 0; i < bufferFilterCount; i++){
        cudaProcessEventsBatchAsync(gpuEventList,gpuEventListSize,
                                    cpuArrGpuFilters[i],fsx,fsy,fsz,
                                    cpuArrGpuConvBuffers[i],ringBufferIdx,
                                    bsx,bsy,bsz,
                                    cudaStreams[i]);
    }
}

long MotionEnergyEstimator::startReadMotionEnergyAsync(float** gpuEnergyBuffers)
{
    eventReadMutex.lock();
    int sxy = bsx*bsy;
    //qDebug("Slots: %d", eventsR->slotsToSkip);
    for(int i = 0; i < orientations.length(); i++){
        // Only syncronize important streams
        cudaStreamSynchronize(cudaStreams[i*FILTERS_PER_ORIENTATION]);
        cudaStreamSynchronize(cudaStreams[i*FILTERS_PER_ORIENTATION + 1]);
        cudaReadMotionEnergyAsync(cpuArrGpuConvBuffers[i*FILTERS_PER_ORIENTATION],
                                          cpuArrGpuConvBuffers[i*FILTERS_PER_ORIENTATION + 1],
                                          ringBufferIdx,
                                          bsx,bsy,
                                          gpuEnergyBuffers[i],
                                          cudaStreams[i*FILTERS_PER_ORIENTATION]);

    }

    for(int i = 0; i < orientations.length(); i++){
        cudaStreamSynchronize(cudaStreams[i*FILTERS_PER_ORIENTATION]);
    }

    // Set skipped slots to zero
    int slotsToSkip = eventsR->slotsToSkip;
    // Split in two operations when we are at the end of the ringbuffer
    if(slotsToSkip + ringBufferIdx > bsz){
        int slotCntOverflow = (slotsToSkip + ringBufferIdx) % bsz;
        //qDebug("Slots to Skip at beginning: %d",slotCntOverflow);
        slotsToSkip -= slotCntOverflow;

        for(int i = 0; i < orientations.length(); i++){
            cudaSetDoubleBuffer(cpuArrGpuConvBuffers[i*FILTERS_PER_ORIENTATION],0,
                    sxy*slotCntOverflow,
                    cudaStreams[i*FILTERS_PER_ORIENTATION]);
            cudaSetDoubleBuffer(cpuArrGpuConvBuffers[i*FILTERS_PER_ORIENTATION + 1],0,
                    sxy*slotCntOverflow,
                    cudaStreams[i*FILTERS_PER_ORIENTATION + 1]);
        }
    }

    if(slotsToSkip > 0){
        slotsToSkip = slotsToSkip % bsz;
        //qDebug("Slots to Skip at end: %d",slotsToSkip);
        for(int i = 0; i < orientations.length(); i++){
            cudaSetDoubleBuffer(cpuArrGpuConvBuffers[i*FILTERS_PER_ORIENTATION] + ringBufferIdx*sxy,0,
                    sxy*slotsToSkip,
                    cudaStreams[i*FILTERS_PER_ORIENTATION]);
            cudaSetDoubleBuffer(cpuArrGpuConvBuffers[i*FILTERS_PER_ORIENTATION + 1] + ringBufferIdx*sxy,0,
                    sxy*slotsToSkip,
                    cudaStreams[i*FILTERS_PER_ORIENTATION + 1]);
        }
    }

    // Go to next timeslice
    long tmp = eventsR->currWindowStartTime;
    ringBufferIdx = (ringBufferIdx+eventsR->slotsToSkip) % cpuArrCpuConvBuffers[0]->getSizeZ();
    eventsR->slotsToSkip = 0;
    eventReadMutex.unlock();
    return tmp;
}
void MotionEnergyEstimator::startNormalizeEnergiesAsync(float** gpuEnergyBuffers)
{
    for(int i = 0; i < orientations.length(); i++){
        // Only syncronize important streams
        cudaStreamSynchronize(cudaStreams[i*FILTERS_PER_ORIENTATION]);
        cudaStreamSynchronize(cudaStreams[i*FILTERS_PER_ORIENTATION + 1]);
        cudaNormalizeMotionEnergyAsync(bsx,bsy,
                                       fsettings.alphaPNorm,fsettings.alphaQNorm,fsettings.betaNorm,fsettings.sigmaBi1,
                                       gpuEnergyBuffers[i],
                                       cudaStreams[i*FILTERS_PER_ORIENTATION]);
    }
}
