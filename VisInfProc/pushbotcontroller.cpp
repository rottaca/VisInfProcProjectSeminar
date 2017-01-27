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

    P = 100;
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
    if(!thread.wait(1000)){
        thread.terminate();
        thread.wait();
    }
}

void PushBotController::startProcessing(){
    processIntervalTimer.start(1000/PUSH_BOT_PROCESS_FPS);
}

void PushBotController::stopProcessing(){
    processIntervalTimer.stop();
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
    // Weighted by their normalized energy

    avgFlowVecXL = 0;
    avgFlowVecYL = 0;
    avgFlowVecXR = 0;
    avgFlowVecYR = 0;
    float cntL = 0,cntR = 0;

    for(int j = 0; j < sx*sy;j++){
        float dir = flowDirPtr[j];
        float s = flowSpeedPtr[j];
        float e = flowEnergyPtr[j];
        if(s > 0){
            // Left or right image half
            if(j % sx < sx/2){
                avgFlowVecXL += cos(dir)*s*e;
                avgFlowVecYL += sin(dir)*s*e;
                cntL+=e;
            }else{
                avgFlowVecXR += cos(dir)*s;
                avgFlowVecYR += sin(dir)*s;
                cntR+=e;
            }
        }
    }

    // Normalize
    avgFlowVecXL /= cntL;
    avgFlowVecYL /= cntL;
    avgFlowVecXR /= cntR;
    avgFlowVecYR /= cntR;

//    qDebug("%f %f", sqrt(avgFlowVecXL*avgFlowVecXL+avgFlowVecYL*avgFlowVecYL),
//           sqrt(avgFlowVecXR*avgFlowVecXR+avgFlowVecYR*avgFlowVecYR));

    float deltaT;
    if(!loopTime.isValid())
        deltaT = 0;
    else{
        deltaT = loopTime.elapsed()/1000.0f;
    }
    loopTime.restart();

    {
        QMutexLocker locker(&pidMutex);
        // Simple PID-Controller
        float e = avgFlowVecXL-avgFlowVecXR;

        eSum = qMax(-eSumMax,qMin(eSum + e,eSumMax));
        out = P*e + I*deltaT*eSum + D/deltaT*(e-eOld);
        eOld = e;
    }
}
