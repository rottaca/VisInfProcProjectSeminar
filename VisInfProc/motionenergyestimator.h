#ifndef MOTIONENERGYESTIMATOR_H
#define MOTIONENERGYESTIMATOR_H

#include "buffer1d.h"
#include "buffer2d.h"
#include "buffer3d.h"
#include "filtermanager.h"
#include "filtersettings.h"
#include "filterset.h"
#include "edvsinterface.h"
#include "settings.h"

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

extern void cudaReadMotionEnergyAsync(float* gpuConvBufferl1,
                                      float* gpuConvBufferl2,
                                      int ringBufferIdx,
                                      int bsx, int bsy,
                                      float* gpuEnergyBuffer,
                                      cudaStream_t cudaStream);
void cudaNormalizeMotionEnergyAsync(int bsx, int bsy,
                                    float alphaPNorm, float alphaQNorm, float betaNorm, float sigmaNorm,
                                    float* gpuEnergyBuffer,
                                    cudaStream_t cudaStream);

#define DEFAULT_STREAM_ID 0

class MotionEnergyEstimator
{
public:
    MotionEnergyEstimator(FilterSettings fs, QVector<float> orientations);
    ~MotionEnergyEstimator();

    FilterSettings getSettings()
    {
        return fsettings;
    }

    void reset();

    bool onNewEvent(const eDVSInterface::DVSEvent &e);
    bool isEventListReady()
    {
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
    quint32 startReadMotionEnergyAsync(float** gpuEnergyBuffers);
    /**
     * @brief startNormalizeEnergiesAsync Normalizes the previously computed motion energy inplace
     * @param gpuEnergyBuffers
     */
    void startNormalizeEnergiesAsync(float** gpuEnergyBuffers);

    /**
     * @brief syncStreams Synchronizes all streams of this motion energy estimator
     */
    void syncStreams()
    {
        for(int i = 0; i < orientations.length()*FILTERS_PER_ORIENTATION; i ++)
            cudaStreamSynchronize(cudaStreams[i]);
    }

    /**
     * @brief getEventsInWindow Returns a vector with all events in the current time window
     * @return
     */
    QList<eDVSInterface::DVSEvent> getEventsInWindow()
    {
        QMutexLocker locker(&eventsInWindowMutex);
        return timeWindowEvents;
    }

    /**
     * @brief getEventStatistics Returns information about the processed and skipped event amount
     * @param all
     * @param skipped
     */
    void getEventStatistics(quint32 &all, quint32 &skipped)
    {
        eventStatisticsMutex.lock();
        all = eventsAll;
        skipped = eventsSkipped;
        eventStatisticsMutex.unlock();
    }

    /**
     * @brief getConvBuffer Debug function to return a copy of the current convolution buffer
     * @param bufferIdx
     * @return
     */
    Buffer3D getConvBuffer(int bufferIdx)
    {
        return *cpuArrCpuConvBuffers[bufferIdx];
    }

    // Struct contains all data for the given list of events
    // -> Number of slots to skip after processing
    // -> The starttime of the current time slot
    // -> All event positions
    typedef struct SlotEventData {
        QVector<SimpleEvent> events;        // Event list (only x and y)
        int slotsToSkip;                    // Timeslots to skip after processing the events
        float currWindowStartTime;         // The start time in us of the current window
    } SlotEventData;

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
    float** cpuArrGpuFilters;
    // Filter sizes
    int fsx,fsy,fsz;

    // Convolution buffers for each filter orientation
    Buffer3D** cpuArrCpuConvBuffers;
    // CPU array of GPU pointers (one pointer for each buffer)
    float** cpuArrGpuConvBuffers;
    // Index into the convolution ring buffer
    int ringBufferIdx;
    // Buffer sizes
    int bsx,bsy,bsz;

    // Overall stream start time TODO: Remove and start stream at 0
    quint32 startTime;
    // The time of the last event; used to detect time jumps
    quint32 lastEventTime;
    // Time per timeslot
    float timePerSlot;

    // All events in the timewindow
    QList<eDVSInterface::DVSEvent> timeWindowEvents;
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

    // Data for statistics computations
    quint32  eventsSkipped;
    quint32 eventsAll;
    QMutex eventStatisticsMutex;
};

#endif // MOTIONENERGYESTIMATOR_H
