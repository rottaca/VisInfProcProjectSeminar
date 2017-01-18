#ifndef PUSHBOTCONTROLLER_H
#define PUSHBOTCONTROLLER_H

#include <QObject>
#include <QThread>
#include <QTimer>
#include <QMutex>
#include <QMutexLocker>
#include <QVector>

#include "settings.h"
#include "filtersettings.h"

class Worker;
class eDVSInterface;

class PushBotController: public QObject
{
    Q_OBJECT
public:
    PushBotController(QObject *parent = 0);
    ~PushBotController();

    void setup(QVector<FilterSettings> settings, QVector<float> orientations){
        QMutexLocker locker(&mutex);
        this->settings = settings;
        this->orientations = orientations;

        flowVecX.resize(settings.length());
        flowVecY.resize(settings.length());
    }

    void setWorker(Worker* worker){
        QMutexLocker locker(&mutex);
        eventProcessor = worker;
    }
    void setRobotInterface(eDVSInterface* interface){
        QMutexLocker locker(&mutex);
        robotInterface = interface;
    }

    void getAvgSpeed(int speedIdx,float &X, float &Y){
        QMutexLocker locker(&mutex);
        X = flowVecX[speedIdx];
        Y = flowVecY[speedIdx];
    }

public slots:
    void startProcessing();
    void stopProcessing();
    void processFlow();

private:
    QThread thread;

    QMutex mutex;
    QTimer processIntervalTimer;
    Worker *eventProcessor;
    eDVSInterface* robotInterface;
    QVector<FilterSettings> settings;
    QVector<float> orientations;
    QVector<float> flowVecX,flowVecY;
};

#endif // PUSHBOTCONTROLLER_H
