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
    isProcessing = true;

    while(isProcessing){
        ofe->process();
        // TODO Busy waiting
        usleep(1);
    }
}

void Worker::nextEvent(DVSEventHandler::DVSEvent event)
{
    ofe->onNewEvent(event);
}
