#ifndef OPTICFLOWESTIMATOR_H
#define OPTICFLOWESTIMATOR_H

#include "motionenergyestimator.h"
#include "edvsinterface.h"

#include <QList>
#include <assert.h>
#include <QVector>

extern void cudaComputeFlow(int sx, int sy,
                            float* gpuEnergy, float* gpuDir, float* gpuSpeed,
                            float** gpuArrGpuEnergies,
                            float* gpuArrOrientations, int orientationCnt,
                            float* gpuArrSpeeds, int speedCnt,
                            float minEnergy,
                            cudaStream_t stream);
extern void cudaFlowToRGB(float* gpuEnergy, float* gpuDir, char* gpuImage,
                          int sx, int sy,
                          float maxLength, cudaStream_t stream);

class OpticFlowEstimator
{
public:
    OpticFlowEstimator(QVector<FilterSettings> settings, QVector<float> orientations);
    ~OpticFlowEstimator();

    /**
     * @brief reset Resets the OpticFlowEstimator to its initial state.
     */
    void reset();

    /**
     * @brief onNewEvent Processes the provided event
     * @param e
     */
    bool onNewEvent(const DVSEvent &e);
    /**
     * @brief process Computes the motion energy in parallel and reads the result
     */
    void process();

    /**
     * @brief getMotionEnergy Provides the raw motion energy for a given orientation and filter number
     * @param filterNr
     * @param orientationIdx
     * @param opponentMotionEnergy Reference to the destination buffer
     * @return Returns the timestamp of the provided data
     */
    quint32 getMotionEnergy(int filterNr, int orientationIdx, Buffer2D &motionEnergy)
    {
        assert(filterNr >= 0);
        assert(filterNr < energyEstimatorCnt);
        assert(orientationIdx >= 0);
        assert(orientationIdx < orientations.length());
        motionEnergyMutex.lock();
        motionEnergy = *(motionEnergyBuffers[filterNr*orientations.length() + orientationIdx]);
        motionEnergyMutex.unlock();
        return updateTimeStamps[filterNr];
    }

    void getOpticFlow(Buffer2D &speed, Buffer2D &dir, Buffer2D &energy)
    {
        motionEnergyMutex.lock();

        if(!opticFlowUpToDate)
            computeOpticFlow();

        speed = opticFlowSpeed;
        dir = opticFlowDir;
        energy = opticFlowEnergy;
        motionEnergyMutex.unlock();
    }

    /**
     * @brief getEventsInWindow Passes the events from the corresonding motion energy estimator to the caller
     * @param filterNr Index of motion energy estimator
     * @return
     */
    QList<DVSEvent> getEventsInWindow(int filterNr)
    {
        assert(filterNr >= 0);
        assert(filterNr < energyEstimatorCnt);
        return motionEnergyEstimators[filterNr]->getEventsInWindow();
    }

    /**
     * @brief getEventStatistics Returns information about the processed and maximal skipped event amount overall
     * @param all
     * @param skipped
     */
    void getEventStatistics( quint32 &all, quint32 &skipped, int filterNr)
    {
        motionEnergyEstimators[filterNr]->getEventStatistics(all,skipped);
    }

    /**
     * @brief getConvBuffer Debug function to return convolution buffer
     * @param filterNr
     * @param orientationIdx
     * @param convBuffer
     */
    void getConvBuffer(int filterNr, int orientationIdx, int pairIdx, Buffer3D &convBuffer)
    {
        assert(filterNr >= 0);
        assert(filterNr < energyEstimatorCnt);
        assert(orientationIdx >= 0);
        assert(orientationIdx < orientations.length());
        motionEnergyMutex.lock();
        convBuffer = motionEnergyEstimators[filterNr]->getConvBuffer(orientationIdx*2+pairIdx);
        motionEnergyMutex.unlock();
    }

    void setEnergyThreshold(float v)
    {
        motionEnergyMutex.lock();
        energyThreshold = v;
        motionEnergyMutex.unlock();
    }

private:
    void computeOpticFlow();

private:
    // Cuda stream for concurrent operations
    cudaStream_t* cudaStreams;
    // Number of created motion energy estimators
    int energyEstimatorCnt;
    // Pointer to array of motion energy estimators
    MotionEnergyEstimator **motionEnergyEstimators;
    // All covered orientations
    QVector<float> orientations;
    float * gpuArrOrientations;
    // All covered filter settings
    QVector<FilterSettings> settings;
    // Contains the combined optic flow
    Buffer2D opticFlowSpeed;
    Buffer2D opticFlowDir;
    Buffer2D opticFlowEnergy;
    float energyThreshold;

    // Mutex for accessing the stored motion energies
    QMutex motionEnergyMutex;
    // Pointer to array of 2d buffer pointers
    Buffer2D **motionEnergyBuffers;
    // CPU array of GPU pointers
    float **cpuArrGpuMotionEnergies;
    // GPU array of GPU pointers
    float **gpuArrGpuMotionEnergies;
    // GPU array of speeds for which the filers are sensitive
    float *gpuArrSpeeds;
    // Array of timestamps of last opponent motion energy updates
    quint32 *updateTimeStamps;
    // True if optic flow is the newest available computation
    bool opticFlowUpToDate;
};

#endif // OPTICFLOWESTIMATOR_H
