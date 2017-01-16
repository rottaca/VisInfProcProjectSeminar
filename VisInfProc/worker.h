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

    void createOpticFlowEstimator(QVector<FilterSettings> settings, QVector<float> orientations);

    void stopProcessing();

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
