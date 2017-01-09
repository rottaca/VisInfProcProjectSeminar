#ifndef OPTICFLOWESTIMATOR_H
#define OPTICFLOWESTIMATOR_H

#include "motionenergyestimator.h"
#include "serialedvsinterface.h"

#include <QList>
#include <assert.h>
#include <QVector>

extern void cudaComputeOpticFlow(int sx, int sy,
                                 float* gpuFlowX, float* gpuFlowY,
                                 float** gpuArrGpuEnergy, float* gpuArrOrientations, int orientationCnt, cudaStream_t stream);

class OpticFlowEstimator
{
public:
    OpticFlowEstimator(QVector<FilterSettings> settings, QVector<float> orientations);
    ~OpticFlowEstimator();

    /**
     * @brief onNewEvent
     * @param e
     */
    void onNewEvent(const SerialeDVSInterface::DVSEvent &e);
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
    long getMotionEnergy(int filterNr, int orientationIdx, Buffer2D &opponentMotionEnergy){
        assert(filterNr >= 0);
        assert(filterNr < energyEstimatorCnt);
        assert(orientationIdx >= 0);
        assert(orientationIdx < orientations.length());
        motionEnergyMutex.lock();
        opponentMotionEnergy = *(opponentMotionEnergies[filterNr*orientations.length() + orientationIdx]);
        motionEnergyMutex.unlock();
        return updateTimeStamps[filterNr];
    }

    /**
     * @brief getOpticFlow Returns the optic flow in x and y direction
     * @param flowX
     * @param flowY
     */
    void getOpticFlow(Buffer2D &flowX, Buffer2D &flowY){

        motionEnergyMutex.lock();
        if(!opticFlowUpToDate)
            computeOpticFlow();

        flowX = opticFlowVec[0];
        flowY = opticFlowVec[1];
        motionEnergyMutex.unlock();
    }

    /**
     * @brief getEventsInWindow Passes the events from the corresonding motion energy estimator to the caller
     * @param filterNr Index of motion energy estimator
     * @return
     */
    QVector<SerialeDVSInterface::DVSEvent> getEventsInWindow(int filterNr){
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

private:
    /**
     * @brief computeOpticFlow Computes the optic flow based on the currently stored information
     */
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
    float * gpuOrientations;
    // All covered filter settings
    QVector<FilterSettings> settings;
    // Contains the optic flow in x and y direction for each pixel
    Buffer2D opticFlowVec[2];
    // Mutex for accessing the stored opponent motion energies
    QMutex motionEnergyMutex;
    // Pointer to array of 2d buffer pointers
    Buffer2D **opponentMotionEnergies;
    // CPU array of GPU pointers
    float **gpuOpponentMotionEnergies;
    // GPU array of GPU pointers
    float **gpuArrgpuOpponentMotionEnergies;
    // Array of timestamps of last opponent motion energy updates
    long *updateTimeStamps;

    bool opticFlowUpToDate;
};

#endif // OPTICFLOWESTIMATOR_H
