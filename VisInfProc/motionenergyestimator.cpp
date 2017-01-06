#include "motionenergyestimator.h"
#include <QtMath>
#include <QElapsedTimer>
#include <QPainter>
#include <assert.h>
#include <iostream>
#include <QFile>
#include "settings.h"
#include <assert.h>

MotionEnergyEstimator::MotionEnergyEstimator(FilterSettings fs, QVector<double> orientations)
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
    bufferFilterCount = orientations.length()*4;
    fset = new FilterSet*[orientations.length()];
    convBuffer = new Buffer3D*[bufferFilterCount];

    gpuFilters = new double* [bufferFilterCount];
    gpuConvBuffers = new double* [bufferFilterCount];
    cudaStreams = new cudaStream_t[bufferFilterCount];

    for(int i = 0; i < orientations.length(); i++){
        fset[i] = new FilterSet(fs,orientations.at(i));
        for(int j = 0; j < 4; j++)
        {
            cudaStreamCreate(&cudaStreams[i*4+j]);
            convBuffer[i*4+j] = new Buffer3D(DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_HEIGHT,fset[i]->sz);
            gpuConvBuffers[i*4+j] = convBuffer[i*4+j]->getGPUPtr();
        }
        gpuFilters[i*4    ] = fset[i]->spatialTemporal[FilterSet::LEFT1].getGPUPtr();
        gpuFilters[i*4 + 1] = fset[i]->spatialTemporal[FilterSet::LEFT2].getGPUPtr();
        gpuFilters[i*4 + 2] = fset[i]->spatialTemporal[FilterSet::RIGHT1].getGPUPtr();
        gpuFilters[i*4 + 3] = fset[i]->spatialTemporal[FilterSet::RIGHT2].getGPUPtr();
    }
    // Extract buffer sizes
    fsx = fset[0]->spatialTemporal[FilterSet::LEFT1].getSizeX();
    fsy = fset[0]->spatialTemporal[FilterSet::LEFT1].getSizeY();
    fsz = fset[0]->spatialTemporal[FilterSet::LEFT1].getSizeZ();
    bsx = DVS_RESOLUTION_WIDTH;
    bsy = DVS_RESOLUTION_HEIGHT;
    bsz = fsz;

    gpuEventList = NULL;
    gpuEventListSize = 0;
}

MotionEnergyEstimator::~MotionEnergyEstimator()
{
    qDebug("Destroying motion energy estimator...");

    for(int i = 0; i < orientations.length(); i++){
        delete fset[i];
        for(int j = 0; j < 4; j++){
            gpuErrchk(cudaStreamDestroy(cudaStreams[i*4+j]));
            delete convBuffer[i*4+j];
        }
    }

    delete[] cudaStreams;
    cudaStreams = NULL;
    delete[] fset;
    fset = NULL;
    delete[] convBuffer;
    convBuffer = NULL;
    delete[] gpuFilters;
    gpuFilters = NULL;
    delete[] gpuConvBuffers;
    gpuConvBuffers = NULL;

    if(gpuEventList != NULL)
        gpuErrchk(cudaFree(gpuEventList));
}
void MotionEnergyEstimator::onNewEvent(const DVSEventHandler::DVSEvent &e){
    eventWriteMutex.lock();

    SimpleEvent ev;
    ev.x = e.posX;
    ev.y = e.posY;

    // Get time from first event as reference
    if(startTime == -1){
        startTime = e.timestamp;
        eventsW->currWindowStartTime = startTime;
    }

    int deltaT = e.timestamp - eventsW->currWindowStartTime;
    // Do we have to skip any timeslots ? Is the new event too new for the current slot ?
    int timeSlotsToSkip = qFloor((double)deltaT/timePerSlot);

    if(timeSlotsToSkip != 0){

        // Flip lists
        eventReadMutex.lock();
        if(eventListReady)
            qDebug("Events skipped: %d",eventsR->events.length());
        SlotEventData* eventsROld = eventsR;
        eventsR = eventsW;
        eventsW = eventsROld;

        eventsR->slotsToSkip=eventsROld->slotsToSkip+timeSlotsToSkip;
        eventsW->slotsToSkip = 0;
        eventsW->currWindowStartTime=eventsR->currWindowStartTime + timePerSlot;

        if(eventsR->events.length() > 0)
            eventListReady = true;

        eventReadMutex.unlock();
        // Clear events for new data
        eventsW->events.clear();
    }
    eventsW->events.append(ev);

    // Update events in timewindow
    eventsInWindowMutex.lock();
    timeWindowEvents.push_back(e);
    while(timeWindowEvents.size() > 0
       && timeWindowEvents.front().timestamp < eventsW->currWindowStartTime - fsettings.timewindow_us)
                    timeWindowEvents.pop_front();
    eventsInWindowMutex.unlock();
    eventWriteMutex.unlock();
}

void MotionEnergyEstimator::startUploadEventsAsync()
{
    eventReadMutex.lock();
    if(eventsR->events.size() > 0)
    {
        // Release previous array
        if(gpuEventList != NULL)
            gpuErrchk(cudaFree(gpuEventList));

        // Allocate buffer for event list
        int cnt = sizeof(SimpleEvent)*eventsR->events.length();
        gpuErrchk(cudaMalloc(&gpuEventList,cnt));
        // Start asynchonous memcpy
        gpuErrchk(cudaMemcpyAsync(gpuEventList,eventsR->events.data(),cnt,cudaMemcpyHostToDevice,cudaStreams[DEFAULT_STREAM_ID]));
        gpuEventListSize = eventsR->events.length();
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
    cudaStreamSynchronize(cudaStreams[DEFAULT_STREAM_ID]);
    assert(gpuEventList != NULL);

    for(int i = 0; i < orientations.length()*4; i++){
        cudaProcessEventsBatchAsync(gpuEventList,gpuEventListSize,
                                    gpuFilters[i],fsx,fsy,fsz,
                                    gpuConvBuffers[i],ringBufferIdx,
                                    bsx,bsy,bsz,
                                    cudaStreams[i]);
    }
}

long MotionEnergyEstimator::startReadMotionEnergyAsync(double** gpuEnergyBuffers)
{
    eventReadMutex.lock();

    for(int i = 0; i < orientations.length(); i++){
        // Only syncronize important streams
        cudaStreamSynchronize(cudaStreams[i*4]);
        cudaStreamSynchronize(cudaStreams[i*4 + 1]);
        cudaStreamSynchronize(cudaStreams[i*4 + 2]);
        cudaStreamSynchronize(cudaStreams[i*4 + 3]);
        cudaReadOpponentMotionEnergyAsync(gpuConvBuffers[i*4],
                                          gpuConvBuffers[i*4 + 1],
                                          gpuConvBuffers[i*4 + 2],
                                          gpuConvBuffers[i*4 + 3],
                                          ringBufferIdx,
                                          bsx,bsy,bsz,
                                          gpuEnergyBuffers[i],
                                          cudaStreams[i*4]);
    }

    // Go to next timeslice
    long tmp = eventsR->currWindowStartTime;
    //qDebug("RingBuffer: %d",ringBufferIdx);
    ringBufferIdx = (ringBufferIdx+eventsR->slotsToSkip) % convBuffer[0]->getSizeZ();
    eventsR->slotsToSkip = 0;
    eventReadMutex.unlock();
    return tmp;
}
