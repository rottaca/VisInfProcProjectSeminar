#ifndef WORKER_H
#define WORKER_H

#include <QObject>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <QSemaphore>

#include "edvsinterface.h"
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
    void setComputationParameters(QVector<FilterSettings> settings, QVector<float> orientations);

    void startProcessing();
    /**
     * @brief stopProcessing Stops the asynchronous event processing
     */
    void stopProcessing();
    /**
     * @brief nextEvent Queues the next event for asynchronous processing
     * @param event
     */
    void nextEvent(const eDVSInterface::DVSEvent &event);

    long getMotionEnergy(int filterNr, int orientationIdx, Buffer2D &opponentMotionEnergy)
    {
        QMutexLocker locker(&mutex);
        if(ofe == NULL)
            return -1;
        long time = ofe->getMotionEnergy(filterNr,orientationIdx,opponentMotionEnergy);
        return time;
    }
    void getOpticFlowEnergy(Buffer2D &energy, Buffer2D &dir, int speedNr)
    {
        QMutexLocker locker(&mutex);
        if(ofe == NULL)
            return;
        ofe->getOpticFlowEnergy(energy,dir,speedNr);
    }
    void getOpticFlow(Buffer2D &speed, Buffer2D &dir, Buffer2D &energy)
    {
        QMutexLocker locker(&mutex);
        if(ofe == NULL)
            return;
        ofe->getOpticFlow(speed,dir,energy);
    }

    void getConvBuffer(int filterNr, int orientationIdx, int pairIdx, Buffer3D &convBuffer)
    {
        QMutexLocker locker(&mutex);
        if(ofe == NULL)
            return;
        ofe->getConvBuffer(filterNr,orientationIdx,pairIdx,convBuffer);
    }

    void getStats(long &recievedEvents, long &dischargedEvents)
    {
        QMutexLocker locker(&mutex);
        if(ofe == NULL)
            return;
        ofe->getEventStatistics(recievedEvents,dischargedEvents);
    }
    void setEnergyThreshold(float v)
    {
        QMutexLocker locker(&mutex);
        if(ofe == NULL)
            return;
        ofe->setEnergyThreshold(v);
    }
    bool isProcessing()
    {
        return processing;
    }

    bool isInitialized()
    {
        return ofe != NULL;
    }

    QList<eDVSInterface::DVSEvent> getEventsInWindow(int filterNr)
    {
        QMutexLocker locker(&mutex);
        if(ofe == NULL)
            return QList<eDVSInterface::DVSEvent>();
        return ofe->getEventsInWindow(filterNr);
    }

    void run();

private:
    bool processing;

    OpticFlowEstimator *ofe;
    QWaitCondition wcWorkReady;
    QVector<FilterSettings> settings;
    QVector<float> orientations;
    QMutex mutex;
};

#endif // WORKER_H
