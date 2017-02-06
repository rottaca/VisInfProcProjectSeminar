#include "edvsinterface.h"
#include <QApplication>
#include <QFile>
#include <QElapsedTimer>

#include "worker.h"
#include "pushbotcontroller.h"


eDVSInterface::eDVSInterface(QObject *parent):QObject(parent)
{
    operationMode = IDLE;
    evBuilderData = NULL;
    evBuilderAddressVersion = Addr2Byte;
    evBuilderTimestampVersion = TimeNoTime;
    evBuilderBufferSz = 0;
    evBuilderByteIdx = 0;
    evBuilderSyncTimestamp = 0;
    processingWorker = NULL;
    pushBotController = NULL;
    playbackSpeed = 1;
    playbackFileName = "";

    moveToThread(&thread);
    socket.moveToThread(&thread);

    // Call process function in eventloop of new thread
    connect(&thread,SIGNAL(started()),this,SLOT(process()));
}

eDVSInterface::~eDVSInterface()
{
    PRINT_DEBUG("Destroying eDVSInterface...");

    operationMutex.lock();
    operationMode = IDLE;
    operationMutex.unlock();

    thread.quit();

    if(!thread.wait(THREAD_WAIT_TIME_MS)) {
        qCritical("Failed to stop eDVSInterface!");
        thread.terminate();
        thread.wait();
    }

    if(evBuilderData != NULL)
        delete[] evBuilderData;
    evBuilderData = NULL;
}

void eDVSInterface::setPushBotCtrl(PushBotController* pushBotCtrl)
{
    QMutexLocker locker(&operationMutex);
    pushBotController = pushBotCtrl;
    connect(this,SIGNAL(onStartPushBotController()),pushBotController,SLOT(startProcessing()));
    connect(this,SIGNAL(onStopPushBotController()),pushBotController,SLOT(stopProcessing()));
}
void eDVSInterface::connectToBot(QString host, int port)
{

    {
        QMutexLocker locker2(&socketMutex);
        this->host = host;
        this->port = port;
    }

    {
        QMutexLocker locker2(&operationMutex);
        operationMode = ONLINE;
    }

    thread.start();
}

void eDVSInterface::startEventStreaming()
{
    {
        QMutexLocker locker(&socketMutex);
        socket.write(CMD_SET_TIMESTAMP_MODE);
        socket.write(CMD_ENABLE_EVENT_STREAMING);
        socket.waitForBytesWritten();
        emit onCmdSent(CMD_SET_TIMESTAMP_MODE);
        emit onCmdSent(CMD_ENABLE_EVENT_STREAMING);
    }
    initEvBuilder(Addr2Byte,TimeDelta);
    {
        QMutexLocker locker(&operationMutex);
        operationMode = ONLINE_STREAMING;
    }
    processingWorker->startProcessing();
    emit onStartPushBotController();
}
void eDVSInterface::stopEventStreaming()
{
    {
        QMutexLocker locker(&operationMutex);
        operationMode = ONLINE;
    }
    {
        QMutexLocker locker(&socketMutex);
        socket.write(CMD_DISABLE_EVENT_STREAMING);
        socket.waitForBytesWritten();
        emit onCmdSent(CMD_DISABLE_EVENT_STREAMING);
    }
    processingWorker->stopProcessing();
    emit onStopPushBotController();
}
void eDVSInterface::sendRawCmd(QString cmd)
{
    if(isConnected()) {
        QMutexLocker locker(&socketMutex);
        socket.write(cmd.toLocal8Bit());
        socket.waitForBytesWritten();
        emit onCmdSent(cmd);
    }
}
void eDVSInterface::enableMotors(bool enable)
{
    if(isConnected()) {
        QMutexLocker locker(&socketMutex);
        if(enable)
            socket.write(CMD_ENABLE_MOTORS);
        else
            socket.write(CMD_DISABLE_MOTORS);
        socket.waitForBytesWritten();
        if(enable)
            emit onCmdSent(CMD_ENABLE_MOTORS);
        else
            emit onCmdSent(CMD_DISABLE_MOTORS);
    }
}
void eDVSInterface::setMotorVelocity(int motorId, int speed)
{
    if(isConnected()) {
        QMutexLocker locker(&socketMutex);
        QString cmd = QString(CMD_SET_VELOCITY).arg(motorId).arg(speed);
        socket.write(cmd.toLocal8Bit());
        socket.waitForBytesWritten();
        emit onCmdSent(cmd);
    }
}
void eDVSInterface::resetBoard()
{
    if(isConnected()) {
        QMutexLocker locker(&socketMutex);
        socket.write(CMD_RESET_BOARD);
        socket.waitForBytesWritten();
        emit onCmdSent(CMD_RESET_BOARD);
    }
}

void eDVSInterface::process()
{

    OperationMode opModeLocal;
    {
        QMutexLocker locker(&operationMutex);
        opModeLocal = operationMode;
    }

    switch (opModeLocal) {
    case PLAYBACK:
        _playbackFile();
        break;
    case ONLINE:
        _processSocket();
        break;
    default:
        break;
    }

    // Process all remaining events
    qApp->processEvents();
    // Wait for processing stopped
    thread.quit();

    PRINT_DEBUG("Done");
}
void eDVSInterface::_processSocket()
{
    // Init serial port
    socket.connectToHost(host,port);
    qInfo("Connecting to %s: %d",host.toLocal8Bit().data(),port);
    if(!socket.waitForConnected(2000)) {
        operationMutex.lock();
        operationMode = IDLE;
        operationMutex.unlock();
        emit onConnectionResult(true);
        qCritical("Can't connect to socket \"%s:%d\": %s"
                  ,host.toLocal8Bit().data(),port,socket.errorString().toLocal8Bit().data());
        return;
    }
    emit onConnectionResult(false);
    PRINT_DEBUG("Connection established");
    OperationMode opModeLocal;
    {
        QMutexLocker locker2(&operationMutex);
        opModeLocal = operationMode;
    }

    DVSEvent eNew;
    quint32 startTimestamp = UINT32_MAX;
    QElapsedTimer timeMeasure;
    char c;

    while(opModeLocal == ONLINE ||
            opModeLocal == ONLINE_STREAMING) {
        {
            QMutexLocker locker(&socketMutex);
            if(socket.state() != QTcpSocket::ConnectedState) {
                PRINT_DEBUG_FMT("Connection closed: %s",socket.errorString().toLocal8Bit().data());
                QMutexLocker locker2(&operationMutex);
                operationMode = IDLE;
                emit onConnectionClosed(true);
                break;
            }

            // Normal command mode
            if(opModeLocal == ONLINE) {
                // Reset streaming time
                if(startTimestamp != UINT32_MAX) {
                    startTimestamp = UINT32_MAX;
                }
                // Wait for command bytes
                if(socket.waitForReadyRead(10)) {
                    // Read data and remove newline
                    QString line = QString(socket.readAll()).remove(QRegExp("[\\n\\t\\r]"));;
                    PRINT_DEBUG_FMT("Recieved: %s",line.toLocal8Bit().data());
                    emit onLineRecived(line);
                }
                // Streaming mode
            } else {
                if(socket.bytesAvailable()) {
                    socket.getChar(&c);
                    if(processingWorker != NULL &&
                            evBuilderProcessNextByte(c,eNew)) {
                        // TODO Check if necessary
                        // send first event directly and start timer
                        if(startTimestamp == UINT32_MAX) {
                            processingWorker->nextEvent(eNew);
                            startTimestamp = eNew.timestamp;
                            timeMeasure.restart();
                        }
                        // Compute sleep time
                        else {
                            quint32 elapsedTimeReal = timeMeasure.nsecsElapsed()/1000;
                            quint32 elapsedTimeEvents = eNew.timestamp - startTimestamp;
                            // Sleep if necessary
                            if(elapsedTimeEvents > elapsedTimeReal) {
                                quint32 sleepTime = elapsedTimeEvents - elapsedTimeReal;
                                PRINT_DEBUG_FMT("Sleep: %u us",sleepTime);
                                QThread::usleep(sleepTime);
                            }

                            processingWorker->nextEvent(eNew);
                        }
                    }
                } else {
                    // TODO Busy waiting bad here ?
                    QThread::usleep(1);
                }
            }
        }

        {
            QMutexLocker locker2(&operationMutex);
            opModeLocal = operationMode;
        }

        qApp->processEvents();
    }
}

void eDVSInterface::stopWork()
{
    operationMutex.lock();
    OperationMode localOpMode = operationMode;
    operationMutex.unlock();

    // Stop streaming
    if(localOpMode == ONLINE_STREAMING)
        stopEventStreaming();

    // Stop processing
    operationMutex.lock();
    operationMode = IDLE;
    operationMutex.unlock();

    PRINT_DEBUG("Stopping eDVSInterface...");
    if(!thread.wait(THREAD_WAIT_TIME_MS)) {
        qCritical("Failed to stop eDVSInterface!");
        thread.terminate();
        thread.wait();
    }
    // Close socket
    if(localOpMode == ONLINE) {
        QMutexLocker locker2(&socketMutex);
        socket.disconnectFromHost();
        emit onConnectionClosed(false);
    }
}

void eDVSInterface::playbackFile(QString fileName, double speed)
{
    {
        QMutexLocker locker(&operationMutex);
        QMutexLocker locker2(&playbackDataMutex);
        if(processingWorker == NULL)
            return;
        operationMode = PLAYBACK;

        playbackFileName = fileName;
        playbackSpeed = speed;
    }
    thread.start();
}
void eDVSInterface::_playbackFile()
{
    double speed;
    QString fileName;
    {
        QMutexLocker locker2(&playbackDataMutex);
        speed = playbackSpeed;
        fileName = playbackFileName;
    }

    Worker* workerLocal;
    {
        QMutexLocker locker(&operationMutex);
        workerLocal = processingWorker;
    }

    if(workerLocal == NULL)
        return;

    PRINT_DEBUG("Start worker");
    workerLocal->startProcessing();
    emit onStartPushBotController();

    // read file, parse header
    TimestampVersion timeVers;
    AddressVersion addrVers;
    QByteArray bytes = parseEventFile(fileName,addrVers,timeVers);
    // Init eventbuilder
    initEvBuilder(addrVers,timeVers);

    OperationMode opModeLocal;
    QElapsedTimer timeMeasure;

    int bufferIdx = 0;
    DVSEvent eNew;
    quint32 startTimestamp = UINT32_MAX;
    int eventCount = 0;
    // Measure real time
    timeMeasure.start();
    do {
        // New event ready ?
        if(evBuilderProcessNextByte(bytes.at(bufferIdx++),eNew)) {
//            if(eNew.On)
//                continue;
            eventCount++;

            // send first event directly
            if(startTimestamp == UINT32_MAX) {
                processingWorker->nextEvent(eNew);
                startTimestamp = eNew.timestamp;
            }
            // Compute sleep time
            else {
                quint32 elapsedTimeReal = timeMeasure.nsecsElapsed()/1000;
                quint32 elapsedTimeEvents = (eNew.timestamp - startTimestamp)/speed;
                // Sleep if necessary
                if(elapsedTimeEvents > elapsedTimeReal) {
                    quint32 sleepTime = elapsedTimeEvents - elapsedTimeReal;
                    QThread::usleep(sleepTime);
                }

                processingWorker->nextEvent(eNew);
            }
        }

        {
            QMutexLocker locker2(&operationMutex);
            opModeLocal = operationMode;
        }
    } while(opModeLocal == PLAYBACK && bufferIdx < bytes.length());

    // Debug info and finished event
    if(bufferIdx == bytes.length()) {
        quint32 elapsedTimeReal = timeMeasure.nsecsElapsed()/1000;
        quint32 elapsedTimeEvents = eNew.timestamp - startTimestamp;
        PRINT_DEBUG_FMT("%s", QString("Executed %1 events in %2 ms instead of %3 ms. Overhead: %4 %")
                        .arg(eventCount)
                        .arg(elapsedTimeReal/1000.0)
                        .arg(elapsedTimeEvents/speed/1000.0)
                        .arg((static_cast<double>(elapsedTimeReal)/(elapsedTimeEvents/speed) - 1) *100)
                        .toLocal8Bit().data());

        {
            QMutexLocker locker2(&operationMutex);
            operationMode = IDLE;
        }
        emit onPlaybackFinished();
    }


    {
        QMutexLocker locker(&operationMutex);
        workerLocal = processingWorker;
    }
    PRINT_DEBUG("Stop worker");
    workerLocal->stopProcessing();
    emit onStopPushBotController();
}

QByteArray eDVSInterface::parseEventFile(QString file, AddressVersion &addrVers, TimestampVersion &timeVers)
{
    QByteArray buff;

    QFile f(file);
    if(!f.open(QIODevice::ReadOnly)) {
        qCritical("Can't open file!");
        return buff;
    }

    buff = f.readAll();
    f.close();

    if(buff.size() == 0) {
        qCritical("File is empty");
        return buff;
    }

    // Parse events from file
    QString versionToken = "#!AER-DAT";
    int versionNr = -1;

    // Parse header
    QList<QByteArray> lines = buff.split('\n');
    int lineIdx =  0;
    for(; lineIdx < lines.length(); lineIdx++) {
        if(!lines.at(lineIdx).startsWith("#"))
            break;
        if(lines.at(lineIdx).contains(versionToken.toLocal8Bit())) {
            QByteArray b = lines.at(lineIdx);
            b.remove(0,versionToken.length());
            b.chop(3);
            versionNr = b.toInt();
        }
    }
    PRINT_DEBUG_FMT("%s", QString("File Version: %1").arg(versionNr).toLocal8Bit().data());

    int dataToSkip = 0;
    for(int i = 0; i < lineIdx; i++)
        dataToSkip += lines.at(i).length()+1;

    PRINT_DEBUG_FMT("%s", QString("Header Size: %1 bytes").arg(dataToSkip).toLocal8Bit().data());

    // Remove header
    buff.remove(0,dataToSkip);

    // Extract events
    int numBytesPerEvent = 6;
    addrVers = Addr2Byte;
    if(versionNr == 2) {
        numBytesPerEvent = 8;
        addrVers = Addr4Byte;
    }
    timeVers = Time4Byte;

    size_t eventCnt = buff.size()/numBytesPerEvent;
    PRINT_DEBUG_FMT("%s", QString("%1 Events.").arg(eventCnt).toLocal8Bit().data());

    return buff;
}

void eDVSInterface::initEvBuilder(AddressVersion addrVers, TimestampVersion timeVers)
{
    //QMutexLocker locker(&evBuilderMutex);
    evBuilderTimestampVersion = timeVers;
    evBuilderAddressVersion = addrVers;
    evBuilderByteIdx = 0;
    evBuilderSyncTimestamp = 0;
    evBuilderLastTimestamp = 0;

    switch(addrVers) {
    case Addr2Byte:
        evBuilderBufferSz = 2;
        break;
    case Addr4Byte:
        evBuilderBufferSz = 4;
    }

    switch(timeVers) {
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

bool eDVSInterface::evBuilderProcessNextByte(char c, DVSEvent &event)
{
    //QMutexLocker locker(&evBuilderMutex);
    // Store byte in buffer
    evBuilderData[evBuilderByteIdx++] = c;
    if(evBuilderTimestampVersion == TimeDelta) {
        // addressbytes done ?
        if(evBuilderByteIdx >= evBuilderAddressVersion) {
            // Check for leading 1 in timestamp bytes
            if(c & 0x80) {
                event = evBuilderParseEvent();
                evBuilderByteIdx = 0;
                return true;
            } else if(evBuilderByteIdx == evBuilderBufferSz) {
                qCritical("Event not recognized! Skipped %d data bytes! "
                          "Please restart!",evBuilderBufferSz);
                evBuilderByteIdx = 0;
            }
        }
    } else {
        // Buffer full ? Event ready
        if(evBuilderByteIdx == evBuilderBufferSz) {
            event = evBuilderParseEvent();
            evBuilderByteIdx = 0;
            return true;
        }
    }
    return false;
}

DVSEvent eDVSInterface::evBuilderParseEvent()
{
    u_int32_t ad = 0,time = 0;
    int idx = 0;
    int addrBytes = evBuilderAddressVersion;
    switch (evBuilderAddressVersion) {
    case Addr2Byte:
        ad |= uint32_t((uchar)evBuilderData[idx++] << 0x08);
        ad |= uint32_t((uchar)evBuilderData[idx++] << 0x00);
        break;
    case Addr4Byte:
        ad = uint32_t((uchar)evBuilderData[idx++] << 0x18);
        ad |= uint32_t((uchar)evBuilderData[idx++] << 0x10);
        ad |= uint32_t((uchar)evBuilderData[idx++] << 0x08);
        ad |= uint32_t((uchar)evBuilderData[idx++] << 0x00);
        break;
    }
    // TODO Use evBuilderSyncTimestamp for all types of timestamps to avoid overflows in time
    switch(evBuilderTimestampVersion) {
    case Time4Byte:
        time = uint32_t((uchar)evBuilderData[idx++] << 0x18);
        time |= uint32_t((uchar)evBuilderData[idx++] << 0x10);
        time |= uint32_t((uchar)evBuilderData[idx++] << 0x08);
        time |= uint32_t((uchar)evBuilderData[idx++] << 0x00);
        // No overflow handling when using 4 bytes
        break;
    case Time3Byte:
        time = uint32_t((uchar)evBuilderData[idx++] << 0x10);
        time |= uint32_t((uchar)evBuilderData[idx++] << 0x08);
        time |= uint32_t((uchar)evBuilderData[idx++] << 0x00);
        // Timestamp overflow ?
        if(time < evBuilderLastTimestamp) {
            evBuilderSyncTimestamp += 0xFFFFFF;
        }
        time += evBuilderSyncTimestamp;
        break;
    case Time2Byte:
        time = uint32_t((uchar)evBuilderData[idx++] << 0x08);
        time |= uint32_t((uchar)evBuilderData[idx++] << 0x00);
        // Timestamp overflow ?
        if(time < evBuilderLastTimestamp) {
            evBuilderSyncTimestamp += 0xFFFF;
        }
        time += evBuilderSyncTimestamp;
        break;
    case TimeDelta: {
        // TODO Check
        // Parse variable timestamp
        // Store bytes in flipped order in time variable
        int pos = (evBuilderByteIdx-1)*7;
        for(int j = 0; j < evBuilderByteIdx-addrBytes; j++) {
            time |= uint32_t(((uchar)evBuilderData[idx++] & 0x7F) << pos);
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

    evBuilderLastTimestamp = time;

    DVSEvent e;
    // Extract event from address by assuming a DVS128 camera
    //e.On = ad & 0x01;       // Polarity: LSB
    // flip axis to match qt's image coordinate system
    e.x = 127 - ((ad >> 0x01) & 0x7F);  // X: 0 - 127
    e.y = 127 - ((ad >> 0x08) & 0x7F) ; // Y: 0 - 127
    e.timestamp = time;

    return e;
}
