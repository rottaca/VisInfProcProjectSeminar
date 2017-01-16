#include "worker.h"

Worker::Worker(QObject *parent) : QThread(parent)
{
    isProcessing = false;
    ofe = NULL;
}


Worker::~Worker(){
    // TODO ABORT WORKER
    if(isProcessing)
        stopProcessing();

    if(ofe != NULL){
        delete ofe;
        ofe = NULL;
    }
}

void Worker::createOpticFlowEstimator(QVector<FilterSettings> settings, QVector<float> orientations)
{
    if(isProcessing)
        stopProcessing();

    if(ofe != NULL)
        delete ofe;

    ofe = new OpticFlowEstimator(settings,orientations);

}

void Worker::stopProcessing()
{
    qDebug("Stopping processing...");
    isProcessing = false;

    if(wait(2000))
        qDebug("Stopped processing.");
    else{
        qDebug("Failed to stop processing thread!");
        terminate();
        wait();
    }
}
void Worker::run()
{
    if(ofe == NULL)
        return;

    isProcessing = true;
    while(isProcessing){
        mutex.lock();
        // Data ready ?
        if(wcWorkReady.wait(&mutex,100))
            ofe->process();
        mutex.unlock();
    }
}

void Worker::nextEvent(const SerialeDVSInterface::DVSEvent &event)
{
    if(ofe == NULL)
        return;

    if(ofe->onNewEvent(event))
        wcWorkReady.wakeAll();

}
