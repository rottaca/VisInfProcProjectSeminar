#ifndef OPTICFLOWESTIMATOR_H
#define OPTICFLOWESTIMATOR_H

#include "motionenergyestimator.h"
#include "edvsinterface.h"

#include <QList>
#include <assert.h>
#include <QVector>

extern void cudaComputeOpticFlow(int sx, int sy,
                                 float* gpuFlowX, float* gpuFlowY,
                                 float** gpuArrGpuEnergy,
                                 float* gpuArrOrientations, int orientationCnt,
                                 float speed,
                                 cudaStream_t stream);
extern void cudaFlowToRGB(float* gpuFlowX, float* gpuFlowY, char* gpuImage,
                            int sx, int sy,
                            float maxLength, cudaStream_t stream);

class OpticFlowEstimator
{
public:
    OpticFlowEstimator(QVector<FilterSettings> settings, QVector<float> orientations);
    ~OpticFlowEstimator();

    /**
     * @brief onNewEvent
     * @param e
     */
    bool onNewEvent(const eDVSInterface::DVSEvent &e);
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
    long getMotionEnergy(int filterNr, int orientationIdx, Buffer2D &motionEnergy){
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
     * @brief getOpticFlow Returns the optic flow in x and y direction
     * @param flowX
     * @param flowY
     */
    void getOpticFlow(Buffer2D &flowX, Buffer2D &flowY, int speedIdx){

        assert(speedIdx >= 0 && speedIdx < energyEstimatorCnt);

        motionEnergyMutex.lock();
        if(!opticFlowUpToDate[speedIdx])
            computeOpticFlow(speedIdx);

        flowX = opticFlowVec[0][speedIdx];
        flowY = opticFlowVec[1][speedIdx];
        motionEnergyMutex.unlock();
    }

    /**
     * @brief getEventsInWindow Passes the events from the corresonding motion energy estimator to the caller
     * @param filterNr Index of motion energy estimator
     * @return
     */
    QVector<eDVSInterface::DVSEvent> getEventsInWindow(int filterNr){
        assert(filterNr >= 0);
        assert(filterNr < energyEstimatorCnt);
        return motionEnergyEstimators[filterNr]->getEventsInWindow();
    }

    /**
     * @brief getEventStatistics Returns information about the processed and maximal skipped event amount overall
     * @param all
     * @param skipped
     */
    void getEventStatistics(long &all, long &skipped){
        long tmp1,tmp2;
        all = 0;
        skipped = 0;
        for(int i = 0; i < energyEstimatorCnt; i++){
            motionEnergyEstimators[i]->getEventStatistics(tmp1,tmp2);
            if(tmp1 > all)
                all = tmp1;
            if(tmp2 > skipped)
                skipped = tmp2;
        }
    }

    /**
     * @brief getConvBuffer Debug function to return convolution buffer
     * @param filterNr
     * @param orientationIdx
     * @param convBuffer
     */
    void getConvBuffer(int filterNr, int orientationIdx, int pairIdx, Buffer3D &convBuffer){
        assert(filterNr >= 0);
        assert(filterNr < energyEstimatorCnt);
        assert(orientationIdx >= 0);
        assert(orientationIdx < orientations.length());
        motionEnergyMutex.lock();
        convBuffer = motionEnergyEstimators[filterNr]->getConvBuffer(orientationIdx*2+pairIdx);
        motionEnergyMutex.unlock();
    }

private:
    /**
     * @brief computeOpticFlow Computes the optic flow based on the currently stored information
     */
    void computeOpticFlow(int speedIdx);

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
    // Contains the optic flow in x and y direction for each pixel
    Buffer2D* opticFlowVec[2];
    // Mutex for accessing the stored opponent motion energies
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
    long *updateTimeStamps;
    // True if optic flow is the newest available computation
    bool *opticFlowUpToDate;
};

#endif // OPTICFLOWESTIMATOR_H
