#include "dvseventhandler.h"
#include <QFile>
#include <QElapsedTimer>
#include "worker.h"

DVSEventHandler::DVSEventHandler(QObject *parent):QThread(parent)
{
    operationMutex.lock();
    operationMode = IDLE;
    operationMutex.unlock();
    playbackSpeed = 1;
    realtimeEventTimestamp = 0;
    worker = NULL;
}
DVSEventHandler::~DVSEventHandler()
{
    operationMutex.lock();
    operationMode = IDLE;
    operationMutex.unlock();

    wait();

    if(realtimeEventData != NULL)
        delete[] realtimeEventData;
}
void DVSEventHandler::run(){

    operationMutex.lock();
    OperationMode opModeLocal = operationMode;
    operationMutex.unlock();

    switch (opModeLocal) {
    case PLAYBACK:
        _playbackFile();
        break;
    case ONLINE:

        break;
    default:
        break;
    }
}

void DVSEventHandler::_playbackFile()
{
    playbackDataMutex.lock();
    QFile f(playbackFileName);
    if(!f.open(QIODevice::ReadOnly)){
        qDebug("Can't open file!");
        return;
    }

    QByteArray buff = f.readAll();
    f.close();
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

    // get new playback speed
    playbackDataMutex.lock();
    float playSpeed = playbackSpeed;
    playbackDataMutex.unlock();

    while(opModeLocal == PLAYBACK && eventIdx < eventList.size()){

        DVSEvent e = eventList.at(eventIdx++);
        worker->nextEvent(e);

        if(eventIdx < eventList.size()){

            int deltaEt = eventList.at(eventIdx).timestamp - startTimestamp;
            int elapsedTime = timeMeasure.nsecsElapsed()/1000*playSpeed;
            int sleepTime = deltaEt - elapsedTime;

            sleepTime = qMax(0,sleepTime);
            if(sleepTime > 0){
                usleep(sleepTime);
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
               arg(dtUs/1000/playSpeed).
               arg(((float)nMillis*1000/dtUs*playSpeed - 1) *100).toLocal8Bit());

        emit onPlaybackFinished();
    }
}

QVector<DVSEventHandler::DVSEvent> DVSEventHandler::parseFile(QByteArray &buff){
    QVector<DVSEvent> events;

    // Parse events from file
    QString versionToken = "#!AER-DAT";
    int versionNr = -1;

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
            versionNr = b.toInt();
        }
    }
    qDebug(QString("Version: %1").arg(versionNr).toLocal8Bit());

    int dataToSkip = 0;
    for(int i = 0; i < lineIdx; i++)
        dataToSkip += lines.at(i).length()+1;

    qDebug(QString("HeaderSize: %1").arg(dataToSkip).toLocal8Bit());

    // Remove header
    buff.remove(0,dataToSkip);

    // Extract events
    int numBytesPerEvent = 6;
    AddressVersion addrVers = Addr2Byte;
    if(versionNr == 2){
         numBytesPerEvent = 8;
         addrVers = Addr4Byte;
    }

    int eventCnt = buff.size()/numBytesPerEvent;
    qDebug(QString("Num Events: %1").arg(eventCnt).toLocal8Bit());
    int buffIdx = 0;

    uchar data[numBytesPerEvent];
    int dataIdx = 0;
    for(int i = 0; i < eventCnt; i++){
        dataIdx = 0;
        for(int j = 0; j < numBytesPerEvent; j++)
            data[dataIdx++] = buff.at(buffIdx++);

        DVSEvent e = parseEvent(data,numBytesPerEvent, addrVers, Time4Byte);

        events.append(e);
    }
    return events;
}

DVSEventHandler::DVSEvent DVSEventHandler::parseEvent(uchar* data, int sz, AddressVersion addrVers, TimestampVersion timeVers)
{
    u_int32_t ad = 0,time = 0;
    int idx = 0;
    int addrBytes = 0;
    switch (addrVers) {
    case Addr2Byte:
        ad |= ((uchar)data[idx++] << 0x08);
        ad |= ((uchar)data[idx++] << 0x00);
        addrBytes =2;
        break;
    case Addr4Byte:
        ad = ((uchar)data[idx++] << 0x18);
        ad |= ((uchar)data[idx++] << 0x10);
        ad |= ((uchar)data[idx++] << 0x08);
        ad |= ((uchar)data[idx++] << 0x00);
        addrBytes =4;
        break;
    }

    switch(timeVers){
    case Time4Byte:
        time = ((uchar)data[idx++] << 0x18);
        time |= ((uchar)data[idx++] << 0x10);
        time |= ((uchar)data[idx++] << 0x08);
        time |= ((uchar)data[idx++] << 0x00);
        break;
    case Time3Byte:
        time |= ((uchar)data[idx++] << 0x10);
        time |= ((uchar)data[idx++] << 0x08);
        time |= ((uchar)data[idx++] << 0x00);
        break;
    case Time2Byte:
        time |= ((uchar)data[idx++] << 0x08);
        time |= ((uchar)data[idx++] << 0x00);
        break;
    case TimeDelta:{
        // TODO Check
        // Parse variable timestamp
        // Store bytes in flipped order in time variable
        int pos = (sz-1)*7;
        for(int j = 0; j < sz-addrBytes; j++){
            time |= (((uchar)data[idx++] & 0x7F) << pos);
            pos-=7;
        }
        break;
    }
    case TimeNoTime:
        time = 0;
        break;
    }

    DVSEvent e;
    // Extract event from address by assuming a DVS128 camera
    e.On = ad & 0x01;       // Polarity: LSB
    // flip axis to match qt's image coordinate system
    e.posX = 127 - ((ad >> 0x01) & 0x7F);  // X: 0 - 127
    e.posY = 127 - ((ad >> 0x08) & 0x7F) ; // Y: 0 - 127
    e.timestamp = time;

    return e;
}

void DVSEventHandler::playbackFile(QString fileName,float speed)
{
    playbackDataMutex.lock();
    playbackFileName = fileName;
    playbackSpeed = speed;
    playbackDataMutex.unlock();

    operationMutex.lock();
    operationMode = PLAYBACK;
    operationMutex.unlock();

    start();
}

void DVSEventHandler::initStreaming(TimestampVersion timeVers)
{
    realtimeDataMutex.lock();
    realtimeEventTimestamp = 0;
    realtimeTimestampVersion = timeVers;
    realtimeEventByteIdx = 0;
    if(realtimeEventData != NULL)
        delete[] realtimeEventData;

    switch(realtimeTimestampVersion){
    case Time4Byte:
        realtimeEventBufferSize = 6;
        break;
    case Time3Byte:
        realtimeEventBufferSize = 5;
        break;
    case Time2Byte:
        realtimeEventBufferSize = 4;
        break;
    case TimeNoTime:
        realtimeEventBufferSize = 2;
        break;
    case TimeDelta:
        realtimeEventBufferSize = 6;
        break;
    }
    realtimeEventData = new uchar[realtimeEventBufferSize];

    realtimeDataMutex.unlock();

    operationMutex.lock();
    operationMode = ONLINE;
    operationMutex.unlock();

    start();
}

void DVSEventHandler::nextRealtimeEventByte(char c)
{
    realtimeDataMutex.lock();

    realtimeEventData[realtimeEventByteIdx++] = c;

    switch(realtimeTimestampVersion){
        case Time2Byte:
        case Time3Byte:
        case Time4Byte:
        case TimeNoTime:
            if(realtimeEventByteIdx == realtimeEventBufferSize){
                worker->nextEvent(parseEvent(realtimeEventData,realtimeEventByteIdx,
                                             Addr2Byte,realtimeTimestampVersion));
                realtimeEventByteIdx = 0;
            }
            break;
        case TimeDelta:
            // minimum 2 bytes for addr
            if(realtimeEventByteIdx > 2){
                // MSB set -> last byte
                if(c & 0x80){
                    DVSEvent e = parseEvent(realtimeEventData,realtimeEventByteIdx,
                                            Addr2Byte,realtimeTimestampVersion);
                    realtimeEventByteIdx = 0;
                    realtimeEventTimestamp += e.timestamp;
                    e.timestamp = realtimeEventTimestamp;
                    worker->nextEvent(e);
                }
            }
            break;
        }

    realtimeDataMutex.unlock();
}

void DVSEventHandler::abort()
{

    qDebug("Stopping playback...");
    operationMutex.lock();
    operationMode = IDLE;
    operationMutex.unlock();

    if(wait(2000))
        qDebug("Stopped playback.");
    else{
        qDebug("Failed to stop playback thread!");
        terminate();
        wait();
    }

    switch (operationMode) {
    case PLAYBACK:

        break;
    case ONLINE:

        break;
    default:
        break;
    }
}
