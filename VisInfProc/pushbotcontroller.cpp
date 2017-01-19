#include "pushbotcontroller.h"

#include "worker.h"
#include "edvsinterface.h"

#include "buffer2d.h"

PushBotController::PushBotController(QObject* parent):QObject(parent)
{
    thread.start();
    moveToThread(&thread);
    processIntervalTimer.moveToThread(&thread);
    eventProcessor = NULL;
    robotInterface = NULL;

    connect(&processIntervalTimer,SIGNAL(timeout()),this,SLOT(processFlow()));
}
PushBotController::~PushBotController()
{
    qDebug("Destroying pushBotController...");
    if(processIntervalTimer.isActive())
        processIntervalTimer.stop();

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
    QMutexLocker locker(&mutex);

    // Get all optic flows
    int speeds = settings.length();

    Buffer2D flowX[speeds], flowY[speeds];

    for(int i = 0; i < speeds; i++){
        eventProcessor->getOpticFlow(flowX[i],flowY[i],i);
    }

    // Compute average flow on left and right image half
    for(int i = 0; i < speeds; i++){
        int sxy = flowX[i].getSizeX()*flowX[i].getSizeY();
        // Compute average flow
        float * fxPtr = flowX[i].getCPUPtr();
        float * fyPtr = flowY[i].getCPUPtr();

        avgFlowVecXL[i] = 0;
        avgFlowVecYL[i] = 0;
        avgFlowVecXR[i] = 0;
        avgFlowVecYR[i] = 0;

        for(int j = 0; j < sxy;j++){
            // Left or right image border
            if(j % flowX[i].getSizeX() < flowX[i].getSizeX()/2){
                avgFlowVecXL[i] += fxPtr[j];
                avgFlowVecYL[i] += fyPtr[j];
            }else{
                avgFlowVecXR[i] += fxPtr[j];
                avgFlowVecYR[i] += fyPtr[j];
            }
        }

        // Normalize
        avgFlowVecXL[i] /= sxy/2;
        avgFlowVecYL[i] /= sxy/2;
        avgFlowVecXR[i] /= sxy/2;
        avgFlowVecYR[i] /= sxy/2;

        // Compute average deviation
        avgFlowDeltaX[i] = avgFlowVecXR[i]-avgFlowVecXR[i];
        avgFlowDeltaY[i] = avgFlowVecYR[i]-avgFlowVecYR[i];
    }
}
