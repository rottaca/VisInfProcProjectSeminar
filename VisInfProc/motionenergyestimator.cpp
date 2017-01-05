#include "motionenergyestimator.h"
#include <QtMath>
#include <QElapsedTimer>
#include <QPainter>
#include <assert.h>
#include <iostream>
#include <QFile>
#include "settings.h"
#include <assert.h>

MotionEnergyEstimator::MotionEnergyEstimator(FilterSettings fs, QList<double> orientations)
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

    double* cpuFilters[bufferFilterCount];
    double* cpuBuffers[bufferFilterCount];

    cudaStreams = new cudaStream_t[orientations.length()];

    for(int i = 0; i < orientations.length(); i++){
        cudaStreamCreate(&cudaStreams[i]);
        fset[i] = new FilterSet(fs,orientations.at(i));
        for(int j = 0; j < 4; j++)
        {
            convBuffer[i*4+j] = new Buffer3D(DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_HEIGHT,fset[i]->sz);
            cpuBuffers[i*4+j] = convBuffer[i*4+j]->getGPUPtr();
        }
        cpuFilters[i*4    ] = fset[i]->spatialTemporal[FilterSet::LEFT1].getGPUPtr();
        cpuFilters[i*4 + 1] = fset[i]->spatialTemporal[FilterSet::LEFT2].getGPUPtr();
        cpuFilters[i*4 + 2] = fset[i]->spatialTemporal[FilterSet::RIGHT1].getGPUPtr();
        cpuFilters[i*4 + 3] = fset[i]->spatialTemporal[FilterSet::RIGHT2].getGPUPtr();
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
    gpuFilters = NULL;
    gpuConvBuffers = NULL;

    int cnt = sizeof(double*)*bufferFilterCount;
    // Allocate buffer for filter pointers
    gpuErrchk(cudaMalloc(&gpuFilters,cnt));
    // Start asynchonous memcpy
    gpuErrchk(cudaMemcpyAsync(gpuFilters,cpuFilters,cnt,cudaMemcpyHostToDevice,cudaStreams[0]));
    // Allocate buffer for buffer pointers
    gpuErrchk(cudaMalloc(&gpuConvBuffers,cnt));
    // Start asynchonous memcpy
    gpuErrchk(cudaMemcpyAsync(gpuConvBuffers,cpuBuffers,cnt,cudaMemcpyHostToDevice,cudaStreams[0]));

}

MotionEnergyEstimator::~MotionEnergyEstimator()
{
    for(int i = 0; i < orientations.length(); i++){
        gpuErrchk(cudaStreamDestroy(cudaStreams[i]));
        delete fset[i];
        for(int j = 0; j < 4; j++){
            delete convBuffer[i*4+j];
        }
    }

    delete[] cudaStreams;
    cudaStreams = NULL;
    delete[] fset;
    fset = NULL;
    delete[] convBuffer;
    convBuffer = NULL;

    if(gpuEventList != NULL)
        gpuErrchk(cudaFree(gpuEventList));

    gpuErrchk(cudaFree(gpuFilters));
    gpuErrchk(cudaFree(gpuConvBuffers));
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
        //qDebug("New Slot finished: %d events",eventsW->events.length());
        // Flip lists
        eventReadMutex.lock();
        SlotEventData* eventsROld = eventsR;
        eventsR = eventsW;
        eventsW = eventsROld;

        eventsR->slotsToSkip=eventsROld->slotsToSkip+timeSlotsToSkip;
        eventsW->slotsToSkip = 0;
        eventsW->currWindowStartTime=eventsR->currWindowStartTime + timePerSlot;
        eventReadMutex.unlock();

        eventsW->events.clear();

        eventListReady = true;
    }
//    else{
//        qDebug("Event in slot %d", e.timestamp);
//    }
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
        //qDebug("Uploading: %d events", eventsR->events.length());
        // Release previous array
        if(gpuEventList != NULL)
            gpuErrchk(cudaFree(gpuEventList));

        // Allocate buffer for event list
        int cnt = sizeof(SimpleEvent)*eventsR->events.length();
        gpuErrchk(cudaMalloc(&gpuEventList,cnt));
        // Start asynchonous memcpy
        gpuErrchk(cudaMemcpyAsync(gpuEventList,eventsR->events.data(),cnt,cudaMemcpyHostToDevice,cudaStreams[0]));
        gpuEventListSize = eventsR->events.length();
        // Free list and wait for next one
        eventsR->events.clear();
    }
    eventReadMutex.unlock();

    eventWriteMutex.lock();
    eventListReady = false;
    eventWriteMutex.unlock();
}

void MotionEnergyEstimator::startProcessEventsBatchAsync()
{
    gpuDataMutex.lock();
    assert(gpuEventList != NULL);

    for(int i = 0; i < orientations.length(); i++){
        cudaProcessEventsBatchAsync(gpuEventList,gpuEventListSize,
                                    &gpuFilters[i*4],fsx,fsy,fsz,
                                    &gpuConvBuffers[i*4],ringBufferIdx,
                                    bsx,bsy,bsz,
                                    cudaStreams[i]);
    }
    gpuDataMutex.unlock();
//    // Events are sorted by time
//    assert(timeWindowEvents.size() == 0 || e.timestamp >= timeWindowEvents.back().timestamp);

//    timeWindowEvents.push_back(e);
//    QVector2D ePos(e.posX,e.posY);

//    int deltaT = (e.timestamp-startTime) - currentSlotStartTime;
//    // Do we have to skip any timeslots ? Is the new event too new for the current slot ?
//    int timeSlotsToSkip = qFloor((float)deltaT/timePerSlot);

//    for(int i = 0; i < orientations.length(); i++){

//        // Skip time slots
//        if(timeSlotsToSkip > 0){
//            conv[i*4+0]->nextTimeSlot(&left1,timeSlotsToSkip);
//            conv[i*4+1]->nextTimeSlot(&left2,timeSlotsToSkip);
//            conv[i*4+2]->nextTimeSlot(&right1,timeSlotsToSkip);
//            conv[i*4+3]->nextTimeSlot(&right2,timeSlotsToSkip);

//            int sx = left1.getSizeX();
//            int sy = left1.getSizeY();

//            cudaComputeOpponentMotionEnergy(sx,sy,
//                                    left1.getGPUPtr(),left2.getGPUPtr(),
//                                    right1.getGPUPtr(),right2.getGPUPtr(),
//                                    opponentMotionEnergy[i].getGPUPtr());

//            isMotionEnergyReady = true;

//            currentSlotStartTime += timePerSlot*timeSlotsToSkip;

//            while(timeWindowEvents.size() > 0
//                  && timeWindowEvents.front().timestamp - startTime < currentSlotStartTime - fsettings.timewindow_us)
//                timeWindowEvents.pop_front();
//        }
//        // Convolute all four filters for this direction
//        conv[i*4+0]->convolute3D(fset[i]->spatialTemporal[FilterSet::LEFT1],ePos);
//        conv[i*4+1]->convolute3D(fset[i]->spatialTemporal[FilterSet::LEFT2],ePos);
//        conv[i*4+2]->convolute3D(fset[i]->spatialTemporal[FilterSet::RIGHT1],ePos);
//        conv[i*4+3]->convolute3D(fset[i]->spatialTemporal[FilterSet::RIGHT2],ePos);
//    }
}

long MotionEnergyEstimator::startReadMotionEnergyAsync(double** gpuEnergyBuffers,int cnt)
{
    eventReadMutex.lock();
    gpuDataMutex.lock();
    assert(cnt*4 == bufferFilterCount);
    cudaReadOpponentMotionEnergyAsync(gpuConvBuffers,bufferFilterCount,ringBufferIdx,
                                      bsx,bsy,bsz,
                                      gpuEnergyBuffers,cnt,
                                      cudaStreams[1]);
    gpuDataMutex.unlock();

    // Go to next timeslice
    long tmp = eventsR->currWindowStartTime;
    //qDebug("RingBuffer: %d",ringBufferIdx);
    ringBufferIdx = (ringBufferIdx+eventsR->slotsToSkip) % convBuffer[0]->getSizeZ();
    eventsR->slotsToSkip = 0;
    eventReadMutex.unlock();
    return tmp;
}
