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

    void createOpticFlowEstimator(QVector<FilterSettings> settings, QVector<double> orientations);

    void stopProcessing();

    void nextEvent(DVSEventHandler::DVSEvent event);

    long getMotionEnergy(int filterNr, int orientationIdx, Buffer2D &opponentMotionEnergy){
        long time = ofe->getMotionEnergy(filterNr,orientationIdx,opponentMotionEnergy);
        return time;
    }
    void getOpticFlow(Buffer2D &flowX, Buffer2D &flowY){
        ofe->getOpticFlow(flowX,flowY);
    }
    void getStats(long &recievedEvents, long &dischargedEvents){
        ofe->getEventStatistics(recievedEvents,dischargedEvents);
    }

    bool getIsProcessing(){
        return isProcessing;
    }

    QVector<DVSEventHandler::DVSEvent> getEventsInWindow(int filterNr){
        QVector<DVSEventHandler::DVSEvent> events;
        events = ofe->getEventsInWindow(filterNr);
        return events;
    }

    void run();

private:
    bool isProcessing;

    OpticFlowEstimator *ofe;

    // Semaphore for reading and writing new event
    QSemaphore *eventSemaphoreR;
    QSemaphore *eventSemaphoreW;
    DVSEventHandler::DVSEvent currEvent;
};

#endif // WORKER_H
