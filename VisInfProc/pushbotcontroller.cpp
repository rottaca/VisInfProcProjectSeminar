#include "pushbotcontroller.h"

#include "worker.h"
#include "edvsinterface.h"


#include <QDateTime>
#include <QtMath>

PushBotController::PushBotController(QObject* parent):QObject(parent)
{
    thread.start();
    moveToThread(&thread);
    processIntervalTimer.moveToThread(&thread);
    eventProcessor = NULL;
    robotInterface = NULL;

    P = 1;
    I = 0;
    D = 0;
    out = 0;
    eOld = 0;
    eSum = 0;
    eSumMax = 0;

    connect(&processIntervalTimer,SIGNAL(timeout()),this,SLOT(processFlow()));
    connect(this,SIGNAL(stopTimer()),&processIntervalTimer,SLOT(stop()));
}
PushBotController::~PushBotController()
{
    qDebug("Destroying pushBotController...");
    if(processIntervalTimer.isActive())
        emit stopTimer();

    thread.quit();
    if(!thread.wait(1000)) {
        thread.terminate();
        thread.wait();
    }
}
void PushBotController::setRobotInterface(eDVSInterface* interface)
{
    QMutexLocker locker(&mutex);
    robotInterface = interface;
    connect(this,SIGNAL(setMotorVelocity(int,int)),robotInterface,SLOT(setMotorVelocity(int,int)));
    connect(this,SIGNAL(enableMotors(bool)),robotInterface,SLOT(enableMotors(bool)));
}
void PushBotController::startProcessing()
{
    qDebug("Start pushbot controller");
    processIntervalTimer.start(1000/PUSH_BOT_PROCESS_FPS);
    emit enableMotors(true);
    emit setMotorVelocity(0,PUSHBOT_VELOCITY_DEFAULT);
    emit setMotorVelocity(1,PUSHBOT_VELOCITY_DEFAULT);
}

void PushBotController::stopProcessing()
{
    qDebug("Stop pushbot controller");
    processIntervalTimer.stop();
    emit setMotorVelocity(1,0);
    emit setMotorVelocity(0,0);
    emit enableMotors(false);
}

void PushBotController::processFlow()
{

#ifdef DEBUG_FLOW_DIR_ENCODE_INTERPOLATION
    return;
#endif
    QMutexLocker locker(&mutex);

    eventProcessor->getOpticFlow(opticFlowSpeed,opticFlowDir,opticFlowEnergy);
    float* flowSpeedPtr = opticFlowSpeed.getCPUPtr();
    float* flowDirPtr = opticFlowDir.getCPUPtr();
    float* flowEnergyPtr = opticFlowEnergy.getCPUPtr();

    int sx = opticFlowSpeed.getSizeX();
    int sy = opticFlowSpeed.getSizeY();

    // Compute average flow on left and right image half
    // Weighted by their normalized energy (to ignore values with low propability)
    avgFlowVecXL = 0;
    avgFlowVecYL = 0;
    avgFlowVecXR = 0;
    avgFlowVecYR = 0;
    float cntL = 0,cntR = 0;

    for(int j = 0; j < sx*sy; j++) {
        float dir = flowDirPtr[j];
        float s = flowSpeedPtr[j];
        float e = flowEnergyPtr[j];
        if(s > 0) {
            // Left or right image half
            if(j % sx < sx/2) {
                avgFlowVecXL += cos(dir)*s*e;
                avgFlowVecYL += sin(dir)*s*e;
                cntL+=e;
            } else {
                avgFlowVecXR += cos(dir)*s*e;
                avgFlowVecYR += sin(dir)*s*e;
                cntR+=e;
            }
        }
    }

    // Normalize
    if(cntL > PUSHBOT_MIN_DETECTION_ENERGY) {
        avgFlowVecXL /= cntL;
        avgFlowVecYL /= cntL;
    } else {
        avgFlowVecXL = 0;
        avgFlowVecYL = 0;
    }
    if(cntR > PUSHBOT_MIN_DETECTION_ENERGY) {
        avgFlowVecXR /= cntR;
        avgFlowVecYR /= cntR;
    } else {
        avgFlowVecXR = 0;
        avgFlowVecYR = 0;
    }

//    qDebug("%f %f", sqrt(avgFlowVecXL*avgFlowVecXL+avgFlowVecYL*avgFlowVecYL),
//           sqrt(avgFlowVecXR*avgFlowVecXR+avgFlowVecYR*avgFlowVecYR));

    float deltaT;
    if(!loopTime.isValid())
        deltaT = 0;
    else {
        deltaT = loopTime.elapsed()/1000.0f;
    }
    loopTime.restart();
    {
        QMutexLocker locker(&pidMutex);
        // Simple PID-Controller
        float e = avgFlowVecXL-avgFlowVecXR;

        eSum = qMax(-eSumMax,qMin(eSum + e,eSumMax));
        // Ignore differential part in first run
        if(deltaT > 0)
            out = P*e + I*deltaT*eSum + D/deltaT*(e-eOld);
        else
            out = P*e + I*deltaT*eSum;

        eOld = e;

        // Send commands to pushbot
        int speedLeft = CLAMP((int)round(PUSHBOT_VELOCITY_DEFAULT - out/2),
                              PUSHBOT_VELOCITY_MIN,PUSHBOT_VELOCITY_MAX);
        int speedRight = CLAMP((int)round(PUSHBOT_VELOCITY_DEFAULT + out/2),
                               PUSHBOT_VELOCITY_MIN,PUSHBOT_VELOCITY_MAX);

        if(robotInterface->isStreaming()) {
            emit setMotorVelocity(PUSHBOT_MOTOR_LEFT,speedLeft);
            emit setMotorVelocity(PUSHBOT_MOTOR_RIGHT,speedRight);
        }
    }
}
