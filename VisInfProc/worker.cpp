#include "worker.h"

Worker::Worker(QObject *parent) : QThread(parent)
{
    processing = false;
    ofe = NULL;
}


Worker::~Worker(){
    qDebug("Destroying worker...");
    if(processing)
        stopProcessing();

    if(ofe != NULL){
        delete ofe;
        ofe = NULL;
    }
}

void Worker::setComputationParameters(QVector<FilterSettings> settings, QVector<float> orientations)
{
    if(processing)
        stopProcessing();

    QMutexLocker locker(&mutex);
    this->settings = settings;
    this->orientations = orientations;
}

void Worker::startProcessing()
{
    if(processing){
        stopProcessing();
    }

    qDebug("Starting processing...");
    {
        QMutexLocker locker(&mutex);
        if(ofe != NULL)
            delete ofe;
        ofe = NULL;
        ofe = new OpticFlowEstimator(settings,orientations);
    }
    qDebug("Started processing.");
    start();
}

void Worker::stopProcessing()
{
    qDebug("Stopping processing...");
    processing = false;

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

    processing = true;
    while(processing){
        mutex.lock();
        // Data ready ?
        if(wcWorkReady.wait(&mutex,100))
            ofe->process();
        mutex.unlock();
    }
}

void Worker::nextEvent(const eDVSInterface::DVSEvent &event)
{
    if(!processing)
        return;

    if(ofe->onNewEvent(event))
        wcWorkReady.wakeAll();
}
