#include "worker.h"
#include <QElapsedTimer>
Worker::Worker(QObject *parent) : QThread(parent)
{
    processing = false;
    ofe = NULL;
}


Worker::~Worker()
{
    PRINT_DEBUG("Destroying worker");

    processing = false;

    if(!wait(THREAD_WAIT_TIME_MS)) {
        qCritical("Failed to stop Worker!");
        terminate();
        wait();
    }
    if(ofe != NULL) {
        delete ofe;
        ofe = NULL;
    }
}

void Worker::setComputationParameters(QVector<FilterSettings> settings, QVector<float> orientations)
{
    if(processing)
        stopProcessing();

    //QMutexLocker locker(&mutex);
    this->settings = settings;
    this->orientations = orientations;

    if(ofe != NULL)
        delete ofe;
    ofe = NULL;
}

void Worker::startProcessing()
{
    if(processing) {
        stopProcessing();
    }

    PRINT_DEBUG("Starting Worker...");
    {
        //QMutexLocker locker(&mutex);
        if(ofe == NULL)
            ofe = new OpticFlowEstimator(settings,orientations);
        else
            ofe->reset();
    }
    PRINT_DEBUG("Worker started.");
    start();
}

void Worker::stopProcessing()
{
    PRINT_DEBUG("Stopping Worker...");
    processing = false;

    if(!wait(THREAD_WAIT_TIME_MS)) {
        qCritical("Failed to stop Worker!");
        terminate();
        wait();
    }
}
void Worker::run()
{
    processing = true;
    while(processing) {
        mutex.lock();
        // Use wait condition to avoid busy waiting
        if(wcWorkReady.wait(&mutex,THREAD_WAIT_TIME_MS/2))
            ofe->process();
        mutex.unlock();
    }
}

void Worker::nextEvent(const DVSEvent &event)
{
    if(!processing)
        return;

    // Notify if work ready
    if(ofe->onNewEvent(event))
        wcWorkReady.wakeAll();
}
