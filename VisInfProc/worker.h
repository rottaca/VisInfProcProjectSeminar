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

    void createOpticFlowEstimator(QList<FilterSettings> settings, QList<double> orientations);

    void stopProcessing();

    void setNextEvent(DVSEventHandler::DVSEvent event);


    void setEnergyThreshold(int filterNr, double threshold){
        ofeMutex.lock();
        ofe->setEnergyThreshold(filterNr,threshold);
        ofeMutex.unlock();
    }

    long getMotionEnergy(int filterNr, int orientationIdx, Buffer2D &opponentMotionEnergy){
        ofeMutex.lock();
        long time = ofe->getMotionEnergy(filterNr,orientationIdx,opponentMotionEnergy);
        ofeMutex.unlock();
        return time;
    }
    void getOpticFlow(Buffer2D &flowX, Buffer2D &flowY){
        ofeMutex.lock();
        ofe->getOpticFlow(flowX,flowY);
        ofeMutex.unlock();
    }
    void getStats(int &recievedEvents, int &dischargedEvents){

        loggingEventMutex.lock();
        recievedEvents = eventCnt;
        dischargedEvents = dischargedEventCnt;
        loggingEventMutex.unlock();
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
