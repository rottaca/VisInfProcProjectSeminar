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

    // Compute average flow
    for(int i = 0; i < speeds; i++){
        // Compute average flow
        float * fxPtr = flowX[i].getCPUPtr();
        float * fyPtr = flowY[i].getCPUPtr();
        flowVecX[i] = 0;
        flowVecY[i] = 0;
        for(int j = 0; j < flowX[i].getSizeX()*flowX[i].getSizeY();j++){
            flowVecX[i] += fxPtr[j];
            flowVecY[i] += fyPtr[j];
        }
    }
}
