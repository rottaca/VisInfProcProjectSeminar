#ifndef OPTICFLOWESTIMATOR_H
#define OPTICFLOWESTIMATOR_H

#include "motionenergyestimator.h"
#include "edvsinterface.h"

#include <QList>
#include <assert.h>
#include <QVector>

extern void cudaComputeFlowEnergyAndDir(int sx, int sy,
                                        float* gpuEnergy, float* gpuDir,
                                        float** gpuArrGpuEnergy,
                                        float* gpuArrOrientations, int orientationCnt,
                                        float speed,
                                        cudaStream_t stream);
extern void cudaFlowToRGB(float* gpuEnergy, float* gpuDir, char* gpuImage,
                          int sx, int sy,
                          float maxLength, cudaStream_t stream);

class OpticFlowEstimator
{
public:
    OpticFlowEstimator(QVector<FilterSettings> settings, QVector<float> orientations);
    ~OpticFlowEstimator();


    void reset();

    /**
     * @brief onNewEvent
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

    /**
     * @brief getOpticFlow Returns the optic flow energy in x and y direction for a given speed idx
     * @param flowX
     * @param flowY
     * @param speedIdx
     */
    void getOpticFlowEnergy(Buffer2D &energy, Buffer2D &dir, int speedIdx)
    {

        assert(speedIdx >= 0 && speedIdx < energyEstimatorCnt);

        motionEnergyMutex.lock();
        if(!opticFlowEnergyUpToDate[speedIdx])
            computeOpticFlowEnergy(speedIdx);

        energy = opticFlowEnergies[speedIdx];
        dir = opticFlowDirs[speedIdx];
        motionEnergyMutex.unlock();
    }

    void getOpticFlow(Buffer2D &speed, Buffer2D &dir, Buffer2D &energy)
    {

        motionEnergyMutex.lock();

        if(!opticFlowUpToDate) {
            for(int i = 0; i < energyEstimatorCnt; i++) {
                if(!opticFlowEnergyUpToDate[i])
                    computeOpticFlowEnergy(i);
            }
            computeOpticFlow();
        }

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
    void getEventStatistics( quint32 &all, quint32 &skipped)
    {
        quint32 tmpAll,tmpSkipped;
        all = 0;
        skipped = 0;
        float ratio = 0, tmpRatio = 0;
        int idx = 0;
        for(int i = 0; i < energyEstimatorCnt; i++) {
            motionEnergyEstimators[i]->getEventStatistics(tmpAll,tmpSkipped);
            tmpRatio = (float)tmpSkipped/tmpAll;
            if(ratio < tmpRatio) {
                all = tmpAll;
                skipped = tmpSkipped;
                ratio = tmpRatio;
                idx = i;
            }
        }
        PRINT_DEBUG_FMT("Speed index most loss: %d",idx);
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
    /**
     * @brief computeOpticFlow Computes the optic flow energy based on the currently stored information
     */
    void computeOpticFlowEnergy(int speedIdx);

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
    // Contains the optic flow in x and y direction for each pixel and for each speed
    Buffer2D* opticFlowEnergies;
    Buffer2D* opticFlowDirs;
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
    // True if optic flow energy is the newest available computation
    bool *opticFlowEnergyUpToDate;
    bool opticFlowUpToDate;
};

#endif // OPTICFLOWESTIMATOR_H
