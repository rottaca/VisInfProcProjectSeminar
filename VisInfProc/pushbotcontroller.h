#ifndef PUSHBOTCONTROLLER_H
#define PUSHBOTCONTROLLER_H

#include <QObject>
#include <QThread>
#include <QTimer>
#include <QTime>
#include <QMutex>
#include <QMutexLocker>
#include <QVector>

#include "settings.h"
#include "filtersettings.h"
#include "buffer2d.h"

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
    }

    void setWorker(Worker* worker){
        QMutexLocker locker(&mutex);
        eventProcessor = worker;
    }
    void setRobotInterface(eDVSInterface* interface);

    void getAvgSpeed(float &XL, float &YL,float &XR, float &YR){
        QMutexLocker locker(&mutex);
        XL = avgFlowVecXL;
        YL = avgFlowVecYL;
        XR = avgFlowVecXR;
        YR = avgFlowVecYR;
    }

    void setP(float _P){
        QMutexLocker locker(&pidMutex);
        P = _P;
    }
    void setI(float _I){
        QMutexLocker locker(&pidMutex);
        I = _I;
    }
    void setD(float _D){
        QMutexLocker locker(&pidMutex);
        D = _D;
    }

    float getCtrlOutput(){
        QMutexLocker locker(&pidMutex);
        return out;
    }

public slots:
    void startProcessing();
    void stopProcessing();
    void processFlow();

signals:
    void stopTimer();
    void setMotorVelocity(int motorId, int speed);
    void enableMotors(bool enable);

private:
    QThread thread;

    QMutex mutex;
    QTimer processIntervalTimer;
    Worker *eventProcessor;
    eDVSInterface* robotInterface;
    QVector<FilterSettings> settings;
    QVector<float> orientations;
    float avgFlowVecXL,avgFlowVecYL;
    float avgFlowVecXR,avgFlowVecYR;

    Buffer2D opticFlowDir,opticFlowSpeed,opticFlowEnergy;

    // Control parameters
    QTime loopTime;
    QMutex pidMutex;
    float P,I,D;
    float out,eOld,eSum, eSumMax;
};

#endif // PUSHBOTCONTROLLER_H
