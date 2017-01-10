#include "serialedvsinterface.h"
#include <QApplication>
#include <QFile>
#include <QElapsedTimer>

#include "worker.h"

SerialeDVSInterface::SerialeDVSInterface(QObject *parent):QObject(parent)
{
    operationMode = IDLE;
    evBuilderData = NULL;
    evBuilderAddressVersion = Addr2Byte;
    evBuilderTimestampVersion = TimeNoTime;
    evBuilderBufferSz = 0;
    evBuilderByteIdx = 0;
    evBuilderSyncTimestamp = 0;
    worker = NULL;
    playbackSpeed = 1;
    playbackFileName = "";

    moveToThread(&thread);
    serial.moveToThread(&thread);

    // Call process function in eventloop of new thread
    connect(&thread,SIGNAL(started()),this,SLOT(process()));

}

SerialeDVSInterface::~SerialeDVSInterface()
{
    operationMutex.lock();
    operationMode = IDLE;
    operationMutex.unlock();

    if(!thread.wait(2000)){
        thread.terminate();
        thread.wait();
    }

    if(evBuilderData != NULL)
        delete[] evBuilderData;
    evBuilderData = NULL;
}

bool SerialeDVSInterface::open(QString portName)
{
    QMutexLocker locker2(&serialMutex);

    // Init serial port
    serial.setBaudRate(4000000);
    serial.setStopBits(QSerialPort::OneStop);
    serial.setDataBits(QSerialPort::Data8);
    serial.setParity(QSerialPort::NoParity);
    serial.setFlowControl(QSerialPort::HardwareControl);
    serial.setPortName(portName);

    if(serial.error() != QSerialPort::NoError){
        qDebug("Serial port error: %d",serial.error());
        return false;
    }

    if(!serial.open(QSerialPort::ReadWrite))
        return false;

    operationMutex.lock();
    operationMode = ONLINE;
    operationMutex.unlock();

    thread.start();
    return true;
}

void SerialeDVSInterface::startEventStreaming()
{
    // TODO
//    QMutexLocker locker2(&dataMutex);
//    if(eventHandler != NULL)
//        eventHandler->initStreaming(DVSEventHandler::TimeDelta);

    QMutexLocker locker(&serialMutex);
    serial.write("E1\n");
    serial.write("E+\n");
    serial.waitForBytesWritten(10);

    operationMutex.lock();
    operationMode = ONLINE_STREAMING;
    operationMutex.unlock();
}
void SerialeDVSInterface::stopEventStreaming()
{
    // TODO
    operationMutex.lock();
    operationMode = ONLINE;
    operationMutex.unlock();

    QMutexLocker locker(&serialMutex);
    serial.write("E-\n");
    serial.waitForBytesWritten(10);
    //QMutexLocker locker2(&dataMutex);
    //if(eventHandler != NULL)
    //    eventHandler->abort();

}
void SerialeDVSInterface::sendRawCmd(QString cmd)
{

    OperationMode opModeLocal;
    {
        QMutexLocker locker(&operationMutex);
        opModeLocal = operationMode;
    }

    if(opModeLocal == ONLINE){
        QMutexLocker locker(&serialMutex);
        serial.write(cmd.toLocal8Bit());
        emit onCmdSent(cmd);
    }
}
void SerialeDVSInterface::enableMotors(bool enable){

    OperationMode opModeLocal;
    {
        QMutexLocker locker(&operationMutex);
        opModeLocal = operationMode;
    }

    if(opModeLocal == ONLINE){
        QMutexLocker locker(&serialMutex);
        if(enable)
            serial.write("M+\n");
        else
            serial.write("M-\n");
        serial.waitForBytesWritten(10);
    }
}
void SerialeDVSInterface::setMotorVelocity(int motorId, int speed)
{
    OperationMode opModeLocal;
    {
        QMutexLocker locker(&operationMutex);
        opModeLocal = operationMode;
    }

    if(opModeLocal == ONLINE){
        QMutexLocker locker(&serialMutex);
        serial.write(QString("MV%1=%2\n").arg(motorId).arg(speed).toLocal8Bit());
        serial.waitForBytesWritten(10);
    }
}

void SerialeDVSInterface::process()
{

    OperationMode opModeLocal;
    {
        QMutexLocker locker(&operationMutex);
        opModeLocal = operationMode;

        if(worker == NULL)
            return;

        qDebug("Start worker");
        worker->start();
    }


    switch (opModeLocal) {
    case PLAYBACK:
        _playbackFile();
        break;
    case ONLINE:

        break;
    default:
        break;
    }

    {
        QMutexLocker locker(&operationMutex);
        qDebug("Stop worker");
        worker->stopProcessing();
    }
    thread.quit();


//    dataMutex.lock();
//    bool openedLocal = opened;
//    bool streamingLocal = streaming;
//    dataMutex.unlock();

//    while(openedLocal){
//        {
//            QMutexLocker locker(&serialMutex);
//            // Normal command mode
//            if(!streamingLocal){
//                if(serial.waitForReadyRead(10) &&
//                        serial.canReadLine()){
//                    QString line = serial.readLine();
//                    emit onLineRecived(line);
//                }
//            // Streaming mode
//            }else{
//                if(serial.waitForReadyRead(10))
//                {
//                    char c;
//                    serial.getChar(&c);
//                    // Event read -> deliver
//                    if(eventHandler != NULL){
//                        eventHandler->nextRealtimeEventByte(c);
//                    }
//                }
//            }
//        }
//        dataMutex.lock();
//        openedLocal = opened;
//        streamingLocal = streaming;
//        dataMutex.unlock();

//        qApp->processEvents();
//    }
}

void SerialeDVSInterface::stop()
{
    if(isStreaming())
        stopEventStreaming();

    operationMutex.lock();
    operationMode = IDLE;
    operationMutex.unlock();

    qDebug("Stopping playback/stream/communication...");
    if(!thread.wait(2000)){
        thread.terminate();
        thread.wait();
        qDebug("Stopped.");
    }
}

void SerialeDVSInterface::playbackFile(QString fileName, float speed)
{
    {
        QMutexLocker locker(&operationMutex);
        QMutexLocker locker2(&playbackDataMutex);
        if(worker == NULL)
            return;
        operationMode = PLAYBACK;

        playbackFileName = fileName;
        playbackSpeed = speed;
    }
    thread.start();
}
void SerialeDVSInterface::_playbackFile()
{
    float speed;
    QString fileName;
    {
        QMutexLocker locker2(&playbackDataMutex);
        speed = playbackSpeed;
        fileName = playbackFileName;
    }
    // read file, parse header
    TimestampVersion timeVers;
    AddressVersion addrVers;
    QByteArray bytes = parseEventFile(fileName,addrVers,timeVers);
    // Init eventbuilder
    initEvBuilder(addrVers,timeVers);

    OperationMode opModeLocal;
    QElapsedTimer timeMeasure;

    long bufferIdx = 0;
    DVSEvent eNew;
    long startTimestamp = -1;
    long eventCount = 0;
    // Measure real time
    timeMeasure.start();
    do{
        // New event ready ?
        if(evBuilderProcessNextByte(bytes.at(bufferIdx++),eNew)){
            eventCount++;

            // send first event directly
            if(startTimestamp == -1){
                worker->nextEvent(eNew);
                startTimestamp = eNew.timestamp;
            }
            // Compute sleep time
            else{
                int elapsedTimeReal = timeMeasure.nsecsElapsed()/1000*speed;
                int elapsedTimeEvents = eNew.timestamp - startTimestamp;
                int sleepTime = elapsedTimeEvents - elapsedTimeReal;

                // Sleep if necessary
                sleepTime = qMax(0,sleepTime);
                if(sleepTime > 0){
                    struct timespec ts = { sleepTime / 1000000, (sleepTime % 1000000) * 1000};
                    nanosleep(&ts, NULL);
                }
                worker->nextEvent(eNew);
            }
        }

        {
            QMutexLocker locker2(&operationMutex);
            opModeLocal = operationMode;
        }
    }while(opModeLocal == PLAYBACK && bufferIdx < bytes.length());

    // Debug info and finished event
    if(bufferIdx == bytes.length()){
        long elapsedTimeReal = timeMeasure.nsecsElapsed()/1000;
        long elapsedTimeEvents = eNew.timestamp - startTimestamp;
        qDebug(QString("Executed %1 events in %2 ms instead of %3 ms. Overhead: %4 %")
               .arg(eventCount)
               .arg(elapsedTimeReal/1000.0f)
               .arg(elapsedTimeEvents/speed/1000.0f)
               .arg(((float)elapsedTimeReal/(elapsedTimeEvents/speed) - 1) *100)
               .toLocal8Bit());

        emit onPlaybackFinished();
    }
}

QByteArray SerialeDVSInterface::parseEventFile(QString file, AddressVersion &addrVers, TimestampVersion &timeVers)
{
    QByteArray buff;

    QFile f(file);
    if(!f.open(QIODevice::ReadOnly)){
        qDebug("Can't open file!");
        return buff;
    }

    buff = f.readAll();
    f.close();

    if(buff.size() == 0){
        qDebug("File is empty");
        return buff;
    }

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
    qDebug(QString("File Version: %1").arg(versionNr).toLocal8Bit());

    int dataToSkip = 0;
    for(int i = 0; i < lineIdx; i++)
        dataToSkip += lines.at(i).length()+1;

    qDebug(QString("HeaderSize: %1").arg(dataToSkip).toLocal8Bit());

    // Remove header
    buff.remove(0,dataToSkip);

    // Extract events
    int numBytesPerEvent = 6;
    addrVers = Addr2Byte;
    if(versionNr == 2){
         numBytesPerEvent = 8;
         addrVers = Addr4Byte;
    }
    timeVers = Time4Byte;

    int eventCnt = buff.size()/numBytesPerEvent;
    qDebug(QString("Num Events: %1").arg(eventCnt).toLocal8Bit());

    return buff;
}

void SerialeDVSInterface::initEvBuilder(AddressVersion addrVers, TimestampVersion timeVers)
{

    QMutexLocker locker(&evBuilderMutex);
    evBuilderTimestampVersion = timeVers;
    evBuilderAddressVersion = addrVers;
    evBuilderByteIdx = 0;
    evBuilderSyncTimestamp = 0;

    switch(addrVers){
    case Addr2Byte:
        evBuilderBufferSz = 2;
        break;
    case Addr4Byte:
        evBuilderBufferSz = 4;
    }

    switch(timeVers){
    case Time4Byte:
        evBuilderBufferSz += 4;
        break;
    case Time3Byte:
        evBuilderBufferSz += 3;
        break;
    case Time2Byte:
        evBuilderBufferSz += 2;
        break;
    case TimeNoTime:
        evBuilderBufferSz += 0;
        break;
    case TimeDelta:
        evBuilderBufferSz += 4;
        break;
    }

    if(evBuilderData != NULL)
        delete[] evBuilderData;
    evBuilderData = new char[evBuilderBufferSz];
}

bool SerialeDVSInterface::evBuilderProcessNextByte(char c, DVSEvent &event)
{
    QMutexLocker locker(&evBuilderMutex);
    // Store byte in buffer
    evBuilderData[evBuilderByteIdx++] = c;

    if(evBuilderTimestampVersion == TimeDelta){
        // addressbytes done ?
        if(evBuilderByteIdx > evBuilderAddressVersion){
            // Check for leading 1 in timestamp bytes
            if(c & 0x80){
                event = evBuilderParseEvent();
                evBuilderByteIdx = 0;
                return true;
            }
        }
    }else{
        // Buffer full ? Event ready
        if(evBuilderByteIdx == evBuilderBufferSz){
            event = evBuilderParseEvent();
            evBuilderByteIdx = 0;
            return true;
        }
    }
    return false;
}

SerialeDVSInterface::DVSEvent SerialeDVSInterface::evBuilderParseEvent()
{
    u_int32_t ad = 0,time = 0;
    int idx = 0;
    int addrBytes = evBuilderAddressVersion;
    switch (evBuilderAddressVersion) {
    case Addr2Byte:
        ad |= ((uchar)evBuilderData[idx++] << 0x08);
        ad |= ((uchar)evBuilderData[idx++] << 0x00);
        break;
    case Addr4Byte:
        ad = ((uchar)evBuilderData[idx++] << 0x18);
        ad |= ((uchar)evBuilderData[idx++] << 0x10);
        ad |= ((uchar)evBuilderData[idx++] << 0x08);
        ad |= ((uchar)evBuilderData[idx++] << 0x00);
        break;
    }

    switch(evBuilderTimestampVersion){
    case Time4Byte:
        time = ((uchar)evBuilderData[idx++] << 0x18);
        time |= ((uchar)evBuilderData[idx++] << 0x10);
        time |= ((uchar)evBuilderData[idx++] << 0x08);
        time |= ((uchar)evBuilderData[idx++] << 0x00);
        break;
    case Time3Byte:
        time = ((uchar)evBuilderData[idx++] << 0x10);
        time |= ((uchar)evBuilderData[idx++] << 0x08);
        time |= ((uchar)evBuilderData[idx++] << 0x00);
        break;
    case Time2Byte:
        time = ((uchar)evBuilderData[idx++] << 0x08);
        time |= ((uchar)evBuilderData[idx++] << 0x00);
        break;
    case TimeDelta:{
        // TODO Check
        // Parse variable timestamp
        // Store bytes in flipped order in time variable
        int pos = (evBuilderByteIdx-1)*7;
        for(int j = 0; j < evBuilderByteIdx-addrBytes; j++){
            time |= (((uchar)evBuilderData[idx++] & 0x7F) << pos);
            pos-=7;
        }
        // Convert relative to absolute timestamp
        evBuilderSyncTimestamp += time;
        time = evBuilderSyncTimestamp;
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
