#ifndef WORKER_H
#define WORKER_H

#include <QObject>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <QSemaphore>

#include "opticflowestimator.h"

class Worker : public QThread
{
    Q_OBJECT
public:
    explicit Worker(QObject *parent = 0);
    ~Worker();

    void createOpticFlowEstimator(QList<FilterSettings> settings, QList<float> orientations);

    void stopProcessing();

    void setNextEvent(DVSEventHandler::DVSEvent event);


    void setEnergyThreshold(int filterNr, double threshold){
        ofeMutex.lock();
        ofe->setEnergyThreshold(filterNr,threshold);
        ofeMutex.unlock();
    }

    long getMotionEnergy(int filterNr, int orientationIdx, Buffer2D &energyLeft, Buffer2D &energyRight){
        ofeMutex.lock();
        long time = ofe->getMotionEnergy(filterNr,orientationIdx,energyLeft,energyRight);
        ofeMutex.unlock();
        return time;
    }

    float getProcessingRatio(){
        loggingEventMutex.lock();
        float p = 0;
        if(eventCnt > 0)
            p = 1 - (float)dischargedEventCnt/eventCnt;
        loggingEventMutex.unlock();
        return p;
    }
    bool getIsProcessing(){
        return isProcessing;
    }

    QVector<DVSEventHandler::DVSEvent> getEventsInWindow(int filterNr){
        QVector<DVSEventHandler::DVSEvent> events;
        ofeMutex.lock();
        events = ofe->getEventsInWindow(filterNr);
        ofeMutex.unlock();
        return events;
    }

    void run();

private:
    bool isProcessing;

    OpticFlowEstimator *ofe;
    QMutex ofeMutex;

    // Semaphore for reading and writing new event
    QSemaphore *eventSemaphoreR;
    QSemaphore *eventSemaphoreW;
    DVSEventHandler::DVSEvent currEvent;

    long eventCnt, dischargedEventCnt;
    QMutex loggingEventMutex;
};

#endif // WORKER_H
