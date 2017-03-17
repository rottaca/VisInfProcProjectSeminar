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

    timePerSlot =(float)fsettings.timewindow_us/fsettings.temporalSteps;
    filterCount = orientations.length()*FILTERS_PER_ORIENTATION;

    gpuEventsX = NULL;
    gpuEventsY = NULL;
    gpuEventListSize = 0;
    gpuEventListSizeAllocated = 0;

    eventsR = new SlotEventData();
    eventsW = new SlotEventData();
    fset = new FilterSet*[orientations.length()];
    cpuArrCpuConvBuffers = new Buffer3D*[filterCount];
    cpuArrGpuFilters = new float* [filterCount];
    cpuArrGpuConvBuffers = new float* [filterCount];
    cudaStreams = new cudaStream_t[filterCount];

    for(int i = 0; i < orientations.length(); i++) {
        fset[i] = new FilterSet(fs,orientations.at(i));
        for(int j = 0; j < FILTERS_PER_ORIENTATION; j++) {
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

    reset();
}

MotionEnergyEstimator::~MotionEnergyEstimator()
{
    PRINT_DEBUG("Destroying motion energy estimator...");
    qDebug("Peak event count per computation: %d",gpuEventListSizeAllocated);
    for(int i = 0; i < orientations.length(); i++) {
        delete fset[i];
        for(int j = 0; j < FILTERS_PER_ORIENTATION; j++) {
            gpuErrchk(cudaStreamDestroy(cudaStreams[i*FILTERS_PER_ORIENTATION+j]));
            delete cpuArrCpuConvBuffers[i*FILTERS_PER_ORIENTATION+j];
        }
    }
    delete eventsR;
    eventsR = NULL;
    delete eventsW;
    eventsW = NULL;
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

    if(gpuEventsX != NULL)
        gpuErrchk(cudaFree(gpuEventsX));
    if(gpuEventsY != NULL)
        gpuErrchk(cudaFree(gpuEventsY));
}
void MotionEnergyEstimator::reset()
{
    eventReadMutex.lock();
    eventsR->currWindowStartTime = 0;
    eventsR->events.clear();
    eventsR->slotsToSkip = 0;
    eventReadMutex.unlock();

    eventWriteMutex.lock();
    eventsW->currWindowStartTime = 0;
    eventsW->events.clear();
    eventsW->slotsToSkip = 0;
    eventListReadyForReading = false;
    eventWriteMutex.unlock();

    startTime = UINT32_MAX;
    ringBufferIdx = 0;

    for(int i = 0; i < orientations.length(); i++) {
        for(int j = 0; j < FILTERS_PER_ORIENTATION; j++) {
            cpuArrCpuConvBuffers[i*FILTERS_PER_ORIENTATION+j]->fill(0);
        }
    }

    eventStatisticsMutex.lock();
    eventsAll = 0;
    eventsSkipped = 0;
    //averagedEventsPerTimeslot = 0;
    eventStatisticsMutex.unlock();

    eventsInWindowMutex.lock();
    timeWindowEvents.clear();
    eventsInWindowMutex.unlock();

    lastEventTime = UINT32_MAX;
}

bool MotionEnergyEstimator::onNewEvent(const DVSEvent &e)
{
    {
        QMutexLocker locker(&eventStatisticsMutex);
        eventsAll++;
        // Compute skip factor to skip single events instead of skipping whole timeslots
        /*if(eventsAll > 0) {
            float avgEventsPerSec = averagedEventsPerTimeslot*1000000.f/timePerSlot;
            float ratio = avgEventsPerSec/FLOW_MAX_EVENTS_PER_SEC;
            if(ratio > 1 && qRound(fmod(eventsAll, ratio)) == 0) {
                eventsSkipped++;
                return false;
            }
        }*/
    }
    QMutexLocker locker(&eventWriteMutex);

    // Get time from first event as reference
    if(startTime == UINT32_MAX) {
        startTime = e.timestamp;
        lastEventTime = startTime;
        eventsW->currWindowStartTime = startTime;
    }
    // Do we have an event with older timestamp?
    else if (lastEventTime > e.timestamp) {
        qCritical("Event is not in order according to timestamp! Restarting Buffers...");
        // TODO
        //reset();
        // TODO Don't throw away the current event
        // Anyway, this issue causes a freeze
        return false;
    }

    // Store last timestamp to detect time jumps
    lastEventTime = e.timestamp;

    // Time between event time and start of time slot
    float deltaT = e.timestamp - eventsW->currWindowStartTime;
    // Do we have to skip any timeslots ? Is the new event too new for the current slot ?
    int timeSlotsToSkip = qFloor(deltaT/timePerSlot);

    if(timeSlotsToSkip > 0) {
        // Flip lists
        eventReadMutex.lock();
        // Was the last block not processed by the worker thread ? Then it is lost
        eventStatisticsMutex.lock();
        if(eventListReadyForReading && eventsR->events.length() > 0) {
            eventsSkipped+=eventsR->events.length();
            PRINT_DEBUG_FMT("[MotionEnergyEstimator] Skipped %d events.",eventsR->events.length());
        }
        /*averagedEventsPerTimeslot = (1-FLOW_SKIPPING_LOW_PASS_FILTER_COEFF)*averagedEventsPerTimeslot
                                    + FLOW_SKIPPING_LOW_PASS_FILTER_COEFF*eventsW->events.size();*/
        eventStatisticsMutex.unlock();

        // Flip read and write lists
        SlotEventData* eventsROld = eventsR;
        eventsR = eventsW;
        eventsW = eventsROld;
        // Aggregate slots to skip
        eventsR->slotsToSkip=eventsROld->slotsToSkip+timeSlotsToSkip;
        eventsW->slotsToSkip = 0;
        // Adjust new window start time
        eventsW->currWindowStartTime=eventsR->currWindowStartTime + timePerSlot*timeSlotsToSkip;
        // Is data for processing ready ?
        if(eventsR->events.length() > 0)
            eventListReadyForReading = true;

        eventReadMutex.unlock();
        // Clear events for new data
        eventsW->events.clear();
    }
    // Add simplified event to new write list
    eventsW->events.append(e);

    return eventListReadyForReading;
}

void MotionEnergyEstimator::uploadEvents()
{
    eventReadMutex.lock();
    if(eventsR->events.size() > 0) {
        int cnt = eventsR->events.length();

        // Do we need more memory for the next event list ?
        if(gpuEventListSizeAllocated < cnt) {
            // Delete old list if it exists
            if(gpuEventsX != NULL) {
                gpuErrchk(cudaFree(gpuEventsX));
            }
            if(gpuEventsY != NULL) {
                gpuErrchk(cudaFree(gpuEventsY));
            }

#ifdef DEBUG_INSERT_PROFILER_MARKS
            nvtxMark("Alloc new Event list");
#endif
            // Allocate buffer for event list on cpu and gpu
            gpuErrchk(cudaMalloc(&gpuEventsX,sizeof(uint8_t)*cnt));
            gpuErrchk(cudaMalloc(&gpuEventsY,sizeof(uint8_t)*cnt));
            cpuEventsX.resize(cnt);
            cpuEventsY.resize(cnt);
            gpuEventListSizeAllocated = cnt;
        }

        // Upload events
#ifdef DEBUG_INSERT_PROFILER_MARKS
        nvtxMark(QString("Upload %1 events").arg(eventsR->events.size()).toStdString().c_str());
#endif
        // Not very nice...
        // Convert interleaved struct data into seperate arrays for x and y
        for(int i = 0; i < cnt; i++) {
            cpuEventsX[i] = eventsR->events.at(i).x;
            cpuEventsY[i] = eventsR->events.at(i).y;
        }


        // Upload event data
        gpuErrchk(cudaMemcpyAsync(gpuEventsX,cpuEventsX.data(),
                                  sizeof(uint8_t)*cnt,cudaMemcpyHostToDevice,
                                  cudaStreams[DEFAULT_STREAM_ID]));
        gpuErrchk(cudaMemcpyAsync(gpuEventsY,cpuEventsY.data(),
                                  sizeof(uint8_t)*cnt,cudaMemcpyHostToDevice,
                                  cudaStreams[DEFAULT_STREAM_ID]));

        // Update events in timewindow
        eventsInWindowMutex.lock();
        // Remove old events
        while(timeWindowEvents.size() > 0
                && timeWindowEvents.front().timestamp < eventsR->currWindowStartTime - fsettings.timewindow_us)
            timeWindowEvents.pop_front();
        // Add new events
        for(DVSEvent &e: eventsR->events)
            timeWindowEvents.push_back(e);
        eventsInWindowMutex.unlock();

        // Wait for uploading to finish
        cudaStreamSynchronize(cudaStreams[DEFAULT_STREAM_ID]);
        gpuEventListSize = cnt;
        // Clear read list
        eventsR->events.clear();
    }
    // Store backup of event struct
    eventsRBackup = *eventsR;
    eventsR->slotsToSkip = 0;

    eventReadMutex.unlock();
    // Event data uploaded, the list is now free
    eventWriteMutex.lock();
    eventListReadyForReading = false;
    eventWriteMutex.unlock();
}

void MotionEnergyEstimator::startProcessEventsBatchAsync()
{
    assert(gpuEventsX != NULL);

    for(int i = 0; i < filterCount; i++) {
        cudaProcessEventsBatchAsync(gpuEventsX,gpuEventsY,gpuEventListSize,
                                    cpuArrGpuFilters[i],fsx,fsy,fsz,
                                    cpuArrGpuConvBuffers[i],ringBufferIdx,
                                    bsx,bsy,bsz,
                                    cudaStreams[i]);
    }
}

quint32 MotionEnergyEstimator::startReadMotionEnergyAsync(float** gpuEnergyBuffers)
{
    int slotsToSkip = eventsRBackup.slotsToSkip;
    quint32 windowStartTime = floor(eventsRBackup.currWindowStartTime);
    int sxy = bsx*bsy;

    // Start reading of all motion energies
    for(int i = 0; i < orientations.length(); i++) {
        cudaStreamSynchronize(cudaStreams[i*FILTERS_PER_ORIENTATION]);
        cudaStreamSynchronize(cudaStreams[i*FILTERS_PER_ORIENTATION + 1]);
        cudaReadMotionEnergyAsync(cpuArrGpuConvBuffers[i*FILTERS_PER_ORIENTATION],
                                  cpuArrGpuConvBuffers[i*FILTERS_PER_ORIENTATION + 1],
                                  ringBufferIdx,
                                  bsx,bsy,
                                  gpuEnergyBuffers[i],
                                  cudaStreams[i*FILTERS_PER_ORIENTATION]);
    }
    // Wait until reading is done
    for(int i = 0; i < orientations.length(); i++) {
        cudaStreamSynchronize(cudaStreams[i*FILTERS_PER_ORIENTATION]);
    }


    // Set skipped slots to zero
    // Split in two operations when we are at the end of the ringbuffer
    if(slotsToSkip >= bsz) {
        // Skip full buffer ? Reset all
        for(int i = 0; i < filterCount; i++) {
            cudaSetDoubleBuffer(cpuArrGpuConvBuffers[i],
                                0, sxy*bsz,
                                cudaStreams[i]);
        }
    } else {
        // Only skip parts of the buffer
        int slotsToSkipRemaining = slotsToSkip;
        if(slotsToSkipRemaining + ringBufferIdx > bsz) {
            int slotCntOverflow = (slotsToSkipRemaining + ringBufferIdx) % bsz;
            slotsToSkipRemaining -= slotCntOverflow;

            for(int i = 0; i < filterCount; i++) {
                cudaSetDoubleBuffer(cpuArrGpuConvBuffers[i],
                                    0, sxy*slotCntOverflow,
                                    cudaStreams[i]);
            }
        }
        // Set slice at beginning to zero if neccessary
        if(slotsToSkipRemaining > 0) {
            for(int i = 0; i < filterCount; i++) {
                cudaSetDoubleBuffer(cpuArrGpuConvBuffers[i] + ringBufferIdx*sxy,
                                    0, sxy*slotsToSkipRemaining,
                                    cudaStreams[i]);
            }
        }
    }

    // Timeslice is cleared, go to next valid timeslot
    ringBufferIdx = (ringBufferIdx+slotsToSkip) % cpuArrCpuConvBuffers[0]->getSizeZ();
    return windowStartTime;
}
void MotionEnergyEstimator::startNormalizeEnergiesAsync(float** gpuEnergyBuffers)
{
    for(int i = 0; i < orientations.length(); i++) {
        // Only syncronize important streams
        cudaStreamSynchronize(cudaStreams[i*FILTERS_PER_ORIENTATION]);
        cudaStreamSynchronize(cudaStreams[i*FILTERS_PER_ORIENTATION + 1]);
        cudaNormalizeMotionEnergyAsync(bsx,bsy,
                                       fsettings.alphaPNorm,fsettings.alphaQNorm,fsettings.betaNorm,fsettings.sigmaBi1,
                                       gpuEnergyBuffers[i],
                                       cudaStreams[i*FILTERS_PER_ORIENTATION]);
    }
}
