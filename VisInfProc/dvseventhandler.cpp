#include "dvseventhandler.h"
#include <QFile>
#include <QElapsedTimer>
#include "worker.h"

DVSEventHandler::DVSEventHandler(QObject *parent):QThread(parent)
{
    operationMutex.lock();
    operationMode = IDLE;
    operationMutex.unlock();
    playbackSpeed = -1;
    worker = NULL;
}
DVSEventHandler::~DVSEventHandler()
{
    operationMutex.lock();
    operationMode = IDLE;
    operationMutex.unlock();
    wait();
}
void DVSEventHandler::run(){

    operationMutex.lock();
    OperationMode opModeLocal = operationMode;
    operationMutex.unlock();

    switch (opModeLocal) {
    case PLAYBACK:
        playbackFile();
        break;
    case ONLINE:

        break;
    default:
        break;
    }

}
void DVSEventHandler::playbackFile()
{
    playbackDataMutex.lock();
    QFile f(playbackFileName);
    if(!f.open(QIODevice::ReadOnly)){
        qDebug("Can't open file!");
        return;
    }

    QByteArray buff = f.readAll();
    f.close();
    int playSpeed = playbackSpeed;
    playbackDataMutex.unlock();

    if(buff.size() == 0){
        qDebug("File is empty");
        return;
    }

    QVector<DVSEvent> eventList = parseFile(buff);

    QElapsedTimer timeMeasure;
    int eventIdx = 0;
    operationMutex.lock();
    OperationMode opModeLocal = operationMode;
    operationMutex.unlock();

    long startTimestamp = eventList.first().timestamp;
    timeMeasure.start();

    while(opModeLocal == PLAYBACK && eventIdx < eventList.size()){

        DVSEvent e = eventList.at(eventIdx++);
        worker->nextEvent(e);

        if(eventIdx < eventList.size()){
            if(playSpeed == -1){
                    int deltaEt = eventList.at(eventIdx).timestamp - startTimestamp;
                    int elapsedTime = timeMeasure.nsecsElapsed()/1000;
                    int sleepTime = deltaEt - elapsedTime;

                    sleepTime = qMax(0,sleepTime);
                    if(sleepTime > 0){
                        usleep(sleepTime);
                    }
            }else if (playSpeed > 0){
                usleep(playSpeed);
            }
        }

        operationMutex.lock();
        opModeLocal = operationMode;
        operationMutex.unlock();
    }

    if(eventIdx == eventList.size()){
        int nMillis = timeMeasure.elapsed();
        int dtUs = eventList.last().timestamp - eventList.first().timestamp;
        qDebug(QString("Executed %1 events in %2 instead of %3 ms. Overhead: %4 %").
               arg(eventList.size()).
               arg(nMillis).
               arg(dtUs/1000).
               arg(((float)nMillis*1000/dtUs - 1) *100).toLocal8Bit());
        //qDebug(QString("Events per second: %1").arg((float)worker->getProcessedEventCnt()/nMillis*1000).toLocal8Bit());
        emit OnPlaybackFinished();
    }
}

QVector<DVSEventHandler::DVSEvent> DVSEventHandler::parseFile(QByteArray &buff){
    QVector<DVSEvent> events;

    // Parse events from file
    QString versionToken = "#!AER-DAT";
    int version = -1;

    // Parse header
    QList<QByteArray> lines = buff.split('\n');
    int lineIdx =  0;
    for(; lineIdx < lines.length(); lineIdx++){
        if(!lines.at(lineIdx).startsWith("#"))
            break;
        if(lines.at(lineIdx).contains(versionToken.toLocal8Bit())){
            QByteArray b = lines.at(lineIdx);
            b.remove(0,versionToken.length());
            b.chop(3);
            version = b.toInt();
        }
        //qDebug(lines.at(lineIdx).toStdString().c_str());
    }
    qDebug(QString("Version: %1").arg(version).toLocal8Bit());

    int dataToSkip = 0;
    for(int i = 0; i < lineIdx; i++)
        dataToSkip += lines.at(i).length()+1;
    qDebug(QString("HeaderSize: %1").arg(dataToSkip).toLocal8Bit());
    // Remove header
    buff.remove(0,dataToSkip);
    // Extract events
    int numBytesPerEvent = 6;
    if(version == 2)
        numBytesPerEvent = 8;
    int eventCnt = buff.size()/numBytesPerEvent;
    qDebug(QString("Num Events: %1").arg(eventCnt).toLocal8Bit());
    int buffIdx = 0;
    for(int i = 0; i < eventCnt; i++){
        u_int32_t ad = 0,time = 0;
        switch(version){
        case 1:
            ad = ((uchar)buff.at(buffIdx++) << 0x08);
            ad |= ((uchar)buff.at(buffIdx++) << 0x00);

            time = ((uchar)buff.at(buffIdx++) << 0x18);
            time |= ((uchar)buff.at(buffIdx++) << 0x10);
            time |= ((uchar)buff.at(buffIdx++) << 0x08);
            time |= ((uchar)buff.at(buffIdx++) << 0x00);

            break;
        case 2:
            ad = ((uchar)buff.at(buffIdx++) << 0x18);
            ad |= ((uchar)buff.at(buffIdx++) << 010);
            ad |= ((uchar)buff.at(buffIdx++) << 0x08);
            ad |= ((uchar)buff.at(buffIdx++) << 0x00);

            time = ((uchar)buff.at(buffIdx++) << 0x18);
            time |= ((uchar)buff.at(buffIdx++) << 0x10);
            time |= ((uchar)buff.at(buffIdx++) << 0x08);
            time |= ((uchar)buff.at(buffIdx++) << 0x00);
            break;
        }
        // Extract event from address by assuming a DVS128 camera
        DVSEvent e;
        e.On = ad & 0x01;       // Polarity: LSB
        e.posX = ((ad >> 0x01) & 0x7F);  // X: 0 - 127
        e.posY = ((ad >> 0x08) & 0x7F) ; // Y: 0 - 127
        e.timestamp = time;

        //if(!e.On)
        //    continue;

        events.append(e);
    }
    return events;
}

void DVSEventHandler::PlayBackFile(QString fileName, int speedus)
{
    playbackDataMutex.lock();
    playbackFileName = fileName;
    playbackSpeed = speedus;
    playbackDataMutex.unlock();

    operationMutex.lock();
    operationMode = PLAYBACK;
    operationMutex.unlock();

    start();
}
