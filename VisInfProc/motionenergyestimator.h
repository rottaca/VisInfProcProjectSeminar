#ifndef MOTIONENERGYESTIMATOR_H
#define MOTIONENERGYESTIMATOR_H

#include "buffer1d.h"
#include "buffer2d.h"
#include "buffer3d.h"
#include "filtermanager.h"
#include "filtersettings.h"
#include "filterset.h"
#include "convolution3d.h"
#include "dvseventhandler.h"

#include <assert.h>

#include <QMutex>

#include <cuda.h>
#include <cuda_runtime.h>
#include <datatypes.h>

// External defined cuda functions
extern void cudaProcessEventsBatchAsync(SimpleEvent* gpuEventList,int gpuEventListSize,
                                        double* gpuFilter, int fsx, int fsy, int fsz,
                                        double* gpuBuffer, int ringBufferIdx,
                                        int bsx, int bsy, int bsz,
                                        cudaStream_t cudaStream);

extern void cudaReadOpponentMotionEnergyAsync(double* gpuConvBufferl1,
                                              double* gpuConvBufferl2,
                                              double* gpuConvBufferr1,
                                              double* gpuConvBufferr2,
                                              int ringBufferIdx,
                                              int bsx, int bsy, int bsz,
                                              double* gpuEnergyBuffer,
                                              cudaStream_t cudaStream);

extern void cudaComputeOpponentMotionEnergy(int sx, int sy,
                                    double* gpul1,double* gpul2,
                                    double* gpur1,double* gpur2,
                                   double* gpuEnergy);

class MotionEnergyEstimator
{
public:
    MotionEnergyEstimator(FilterSettings fs, QList<double> orientations);
    ~MotionEnergyEstimator();

    FilterSettings getSettings(){
        return fsettings;
    }


    void onNewEvent(const DVSEventHandler::DVSEvent &e);
    bool isEventListReady(){
        eventWriteMutex.lock();
        bool cpy = eventListReady;
        eventWriteMutex.unlock();
        return cpy;
    }

    void startUploadEventsAsync();
    void startProcessEventsBatchAsync();
    long startReadMotionEnergyAsync(double** gpuEnergyBuffers);

    void syncStreams(){
        for(int i = 0; i < orientations.length()*4; i ++)
            cudaStreamSynchronize(cudaStreams[i]);
    }

    QVector<DVSEventHandler::DVSEvent> getEventsInWindow(){
        eventsInWindowMutex.lock();
        // TODO
        QVector<DVSEventHandler::DVSEvent> tmp = QVector<DVSEventHandler::DVSEvent>(timeWindowEvents);
        eventsInWindowMutex.unlock();
        return tmp;
    }

    // Struct contains all data for the given list of events
    // -> Number of slots to skip after processing
    // -> The starttime of the current time slot
    // -> All event positions
    typedef struct SlotEventData{
        QVector<SimpleEvent> events;
        int slotsToSkip;
        long currWindowStartTime;
    }SlotEventData;

private:
    // Stream for concurrent execution
    cudaStream_t* cudaStreams;

    // Amount of convolution filters and buffers (4*orientationCnt)
    int bufferFilterCount;

    // All orientations
    QList<double> orientations;
    // Filtersettings for the filter
    FilterSettings fsettings;
    // filterset for each orientation of the specified filter
    FilterSet** fset;
    double** gpuFilters;
    int fsx,fsy,fsz;

    // Convolution buffers for each filter orientation
    Buffer3D** convBuffer;
    double** gpuConvBuffers;
    // Index into the convolution ring buffer
    int ringBufferIdx;
    int bsx,bsy,bsz;

    QMutex gpuDataMutex;

    // Starttime for the current time slot
    //long currWindowStartTime;
    // Overall stream start time TODO: Remove and start stream at 0
    int startTime;
    // Time per timeslot
    double timePerSlot;

    //QMutex sharedCpuDataMutex;
    //int slotsToSkip;

    // All events in the timewindow
    QVector<DVSEventHandler::DVSEvent> timeWindowEvents;
    QMutex eventsInWindowMutex;
    // All events in the current timeslot
    SlotEventData timeSlotEvents[2];
    // Pointer to both time slot event lists
    SlotEventData* eventsR, *eventsW;
    // Mutex for accessing event list write ptr
    QMutex eventWriteMutex;
    // Mutex for accessing event list read ptr
    QMutex eventReadMutex;
    // Gpu ptr for eventsR
    SimpleEvent* gpuEventList;
    int gpuEventListSize;
    bool eventListReady;

    QVector<int> eventCnt;
};

#endif // MOTIONENERGYESTIMATOR_H
