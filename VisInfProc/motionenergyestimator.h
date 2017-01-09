#ifndef MOTIONENERGYESTIMATOR_H
#define MOTIONENERGYESTIMATOR_H

#include "buffer1d.h"
#include "buffer2d.h"
#include "buffer3d.h"
#include "filtermanager.h"
#include "filtersettings.h"
#include "filterset.h"
#include "convolution3d.h"
#include "serialedvsinterface.h"

#include <assert.h>

#include <QMutex>

#include <cuda.h>
#include <cuda_runtime.h>
#include <datatypes.h>

// External defined cuda functions
extern void cudaProcessEventsBatchAsync(SimpleEvent* gpuEventList,int gpuEventListSize,
                                        float* gpuFilter, int fsx, int fsy, int fsz,
                                        float* gpuBuffer, int ringBufferIdx,
                                        int bsx, int bsy, int bsz,
                                        cudaStream_t cudaStream);

extern void cudaReadOpponentMotionEnergyAsync(float* gpuConvBufferl1,
                                              float* gpuConvBufferl2,
                                              float* gpuConvBufferr1,
                                              float* gpuConvBufferr2,
                                              int ringBufferIdx,
                                              int bsx, int bsy, int bsz,
                                              float* gpuEnergyBuffer,
                                              cudaStream_t cudaStream);

#define DEFAULT_STREAM_ID 0

class MotionEnergyEstimator
{
public:
    MotionEnergyEstimator(FilterSettings fs, QVector<float> orientations);
    ~MotionEnergyEstimator();

    FilterSettings getSettings(){
        return fsettings;
    }


    void onNewEvent(const SerialeDVSInterface::DVSEvent &e);
    bool isEventListReady(){
        eventWriteMutex.lock();
        bool cpy = eventListReady;
        eventWriteMutex.unlock();
        return cpy;
    }
    /**
     * @brief startUploadEventsAsync Starts the async uploading of the new event list into gpu memory
     */
    void startUploadEventsAsync();
    /**
     * @brief startProcessEventsBatchAsync Starts the async processing of events (convolution)
     *        after syncronizing related streams
     */
    void startProcessEventsBatchAsync();
    /**
     * @brief startReadMotionEnergyAsync Starts the async computation and reading of the motion energy.
     *        The resulting energy for every direction is stored in the provided buffers.
     * @param gpuEnergyBuffers cpu Pointer to an cpu array of gpu buffer pointers (amount == orientations.length()
     * @return Returns the start time for the current time slot
     */
    long startReadMotionEnergyAsync(float** gpuEnergyBuffers);

    /**
     * @brief syncStreams Synchronizes all streams of this motion energy estimator
     */
    void syncStreams(){
        for(int i = 0; i < orientations.length()*4; i ++)
            cudaStreamSynchronize(cudaStreams[i]);
    }

    /**
     * @brief getEventsInWindow Returns a vector with all events in the current time window
     * @return
     */
    QVector<SerialeDVSInterface::DVSEvent> getEventsInWindow(){
        eventsInWindowMutex.lock();
        QVector<SerialeDVSInterface::DVSEvent> tmp = QVector<SerialeDVSInterface::DVSEvent>(timeWindowEvents);
        eventsInWindowMutex.unlock();
        return tmp;
    }

    /**
     * @brief getEventStatistics Returns information about the processed and skipped event amount
     * @param all
     * @param skipped
     */
    void getEventStatistics(long &all, long &skipped){
        eventStatisticsMutex.lock();
        all = eventsAll;
        skipped = eventsSkipped;
        eventStatisticsMutex.unlock();
    }

    // Struct contains all data for the given list of events
    // -> Number of slots to skip after processing
    // -> The starttime of the current time slot
    // -> All event positions
    typedef struct SlotEventData{
        QVector<SimpleEvent> events;        // Event list (only x and y)
        int slotsToSkip;                    // Timeslots to skip after processing the events
        float currWindowStartTime;         // The start time in us of the current window
    }SlotEventData;

private:
    // Stream for concurrent execution
    cudaStream_t* cudaStreams;

    // Amount of convolution filters and buffers (4*orientationCnt)
    int bufferFilterCount;

    // All orientations
    QVector<float> orientations;
    // Filtersettings for the filter
    FilterSettings fsettings;
    // filterset for each orientation of the specified filter
    FilterSet** fset;
    // CPU array of GPU pointers (one pointer for each filter)
    float** gpuFilters;
    // Filter sizes
    int fsx,fsy,fsz;

    // Convolution buffers for each filter orientation
    Buffer3D** convBuffer;
    // CPU array of GPU pointers (one pointer for each buffer)
    float** gpuConvBuffers;
    // Index into the convolution ring buffer
    int ringBufferIdx;
    // Buffer sizes
    int bsx,bsy,bsz;

    // Overall stream start time TODO: Remove and start stream at 0
    int startTime;
    // Time per timeslot
    float timePerSlot;

    // All events in the timewindow
    QVector<SerialeDVSInterface::DVSEvent> timeWindowEvents;
    // Mutex for events in timewindow
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
    // Size of the uploaded event list (Event count)
    int gpuEventListSize;
    // Size of the allocated memory (Event count), increases over time to avoid reallocation for smaller blocks
    int gpuEventListSizeAllocated;
    // True, if a new event list is ready for processing
    bool eventListReady;

    long eventsSkipped;
    long eventsAll;
    QMutex eventStatisticsMutex;
};

#endif // MOTIONENERGYESTIMATOR_H
