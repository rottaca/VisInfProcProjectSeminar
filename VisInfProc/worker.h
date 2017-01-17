#ifndef WORKER_H
#define WORKER_H

#include <QObject>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <QSemaphore>

#include "serialedvsinterface.h"
#include "opticflowestimator.h"

class Worker : public QThread
{
    Q_OBJECT
public:
    explicit Worker(QObject *parent = 0);
    ~Worker();

    /**
     * @brief createOpticFlowEstimator Creates an opticFlow estimator object with the specified filter settings
     * @param settings
     * @param orientations
     */
    void createOpticFlowEstimator(QVector<FilterSettings> settings, QVector<float> orientations);
    /**
     * @brief stopProcessing Stops the asynchronous event processing
     */
    void stopProcessing();
    /**
     * @brief nextEvent Queues the next event for asynchronous processing
     * @param event
     */
    void nextEvent(const SerialeDVSInterface::DVSEvent &event);

    long getMotionEnergy(int filterNr, int orientationIdx, Buffer2D &opponentMotionEnergy){
        long time = ofe->getMotionEnergy(filterNr,orientationIdx,opponentMotionEnergy);
        return time;
    }
    void getOpticFlow(Buffer2D &flowX, Buffer2D &flowY, int speedNr){
        ofe->getOpticFlow(flowX,flowY,speedNr);
    }
    void getConvBuffer(int filterNr, int orientationIdx, int pairIdx, Buffer3D &convBuffer){
        ofe->getConvBuffer(filterNr,orientationIdx,pairIdx,convBuffer);
    }

    void getStats(long &recievedEvents, long &dischargedEvents){
        ofe->getEventStatistics(recievedEvents,dischargedEvents);
    }
    /**
     * @brief getIsProcessing Returns true, when the worker thread is active
     * @return
     */
    bool getIsProcessing(){
        return isProcessing;
    }

    QVector<SerialeDVSInterface::DVSEvent> getEventsInWindow(int filterNr){
        QVector<SerialeDVSInterface::DVSEvent> events;
        events = ofe->getEventsInWindow(filterNr);
        return events;
    }

    void run();

private:
    bool isProcessing;

    OpticFlowEstimator *ofe;
    QWaitCondition wcWorkReady;
    QMutex mutex;
};

#endif // WORKER_H
