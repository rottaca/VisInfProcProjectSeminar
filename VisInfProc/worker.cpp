#include "worker.h"

Worker::Worker(QObject *parent) : QThread(parent)
{
    isProcessing = false;
    eventSemaphoreR = new QSemaphore(0);
    eventSemaphoreW = new QSemaphore(1);
    ofe = NULL;
    eventCnt = 0;
    dischargedEventCnt = 0;
}


Worker::~Worker(){
    // TODO ABORT WORKER
    isProcessing = false;
    wait();
    if(ofe != NULL){
        delete ofe;
        ofe = NULL;
    }
    delete eventSemaphoreR;
    eventSemaphoreR = NULL;
    delete eventSemaphoreW;
    eventSemaphoreW = NULL;
}

void Worker::createOpticFlowEstimator(QList<FilterSettings> settings, QList<double> orientations)
{
    ofeMutex.lock();
    if(ofe != NULL)
        delete ofe;

    ofe = new OpticFlowEstimator(settings,orientations);
    ofeMutex.unlock();

    loggingEventMutex.lock();
    eventCnt = 0;
    dischargedEventCnt = 0;
    loggingEventMutex.unlock();
    // Save ??
    delete eventSemaphoreR;
    eventSemaphoreR = new QSemaphore(0);
    delete eventSemaphoreW;
    eventSemaphoreW = new QSemaphore(1);

}

void Worker::stopProcessing()
{
    qDebug("Stopping processing..");
    isProcessing = false;

    if(wait(2000))
        qDebug("Stopped processing.");
    else
        qDebug("Failed to stop processing thread!");
}
void Worker::run()
{
    eventCnt = 0;
    dischargedEventCnt = 0;
    isProcessing = true;

    while(isProcessing){
        // Try to lock event, don't block forever when we try to stop the processing
        if(!eventSemaphoreR->tryAcquire(1,100))
            continue;

        // Copy event and release it
        DVSEventHandler::DVSEvent e = currEvent;
        eventSemaphoreW->release(1);

        // Process the event
        ofeMutex.lock();
        ofe->processEvent(e);
        ofeMutex.unlock();
    }
}

void Worker::setNextEvent(DVSEventHandler::DVSEvent event)
{
    loggingEventMutex.lock();
    eventCnt++;

    if(!eventSemaphoreW->tryAcquire(1)){
        dischargedEventCnt++;
        loggingEventMutex.unlock();
        return;
    }
    loggingEventMutex.unlock();

    currEvent = event;
    eventSemaphoreR->release(1);
}
