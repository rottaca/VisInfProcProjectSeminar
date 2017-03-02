#include "edvsinterface.h"
#include <QApplication>
#include <QFile>
#include <QElapsedTimer>

#include "worker.h"
#include "pushbotcontroller.h"


eDVSInterface::eDVSInterface(QObject *parent):QObject(parent)
{
    operationMode = IDLE;
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
    thread.quit();

    if(!thread.wait(THREAD_WAIT_TIME_MS)) {
        qCritical("Failed to stop eDVSInterface!");
        thread.terminate();
        thread.wait();
    }

}

void eDVSInterface::setPushBotCtrl(PushBotController* pushBotCtrl)
{
    QMutexLocker locker(&operationMutex);
    pushBotController = pushBotCtrl;
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
    {
        QMutexLocker locker(&operationMutex);
        operationMode = START_STREAMING;
    }
    evBuilder.initEvBuilder(EventBuilder::Addr2Byte,EventBuilder::Time4Byte);
    processingWorker->startProcessing();
}
void eDVSInterface::stopEventStreaming()
{
    {
        QMutexLocker locker(&socketMutex);
        socket.write(CMD_DISABLE_EVENT_STREAMING);
        socket.waitForBytesWritten();
        emit onCmdSent(CMD_DISABLE_EVENT_STREAMING);
    }
    {
        QMutexLocker locker(&operationMutex);
        operationMode = STOP_STREAMING;
    }
    processingWorker->stopProcessing();
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
    qInfo("Connection established");
    OperationMode opModeLocal;
    {
        QMutexLocker locker2(&operationMutex);
        opModeLocal = operationMode;
    }

    DVSEvent eNew;
    quint32 startTimestamp = UINT32_MAX;
    QElapsedTimer timeMeasure;
    char c;

    while(opModeLocal != IDLE) {
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
            if(opModeLocal == ONLINE ||
                    opModeLocal == START_STREAMING ||
                    opModeLocal == STOP_STREAMING) {
                // Reset streaming time
                if(startTimestamp != UINT32_MAX) {
                    startTimestamp = UINT32_MAX;
                }
                // Wait for command bytes
                if(socket.waitForReadyRead(10)) {
                    // Read data and remove newline
                    // .remove(QRegExp("[\\n\\t\\r]"));
                    QString line = QString(socket.readLine());
                    PRINT_DEBUG_FMT("Recieved: %s",line.toLocal8Bit().data());

                    if(opModeLocal == ONLINE) {
                        emit onLineRecived(line);
                    } else if(opModeLocal == START_STREAMING &&
                              line.contains(CMD_ENABLE_EVENT_STREAMING)) {
                        {
                            QMutexLocker locker2(&operationMutex);
                            operationMode = STREAMING;
                        }
                        emit onStreamingStarted();
                        qDebug("Started Streaming");
                    } else if(opModeLocal == STOP_STREAMING &&
                              line.contains(CMD_DISABLE_EVENT_STREAMING)) {
                        {
                            QMutexLocker locker2(&operationMutex);
                            operationMode = ONLINE;
                        }
                        emit onStreamingStopped();
                        qDebug("Stopped Streaming");
                    }
                }
                // Streaming mode
            } else {
                if(socket.bytesAvailable()) {
                    socket.getChar(&c);
                    if(processingWorker != NULL &&
                            evBuilder.evBuilderProcessNextByte(c,eNew,true)) {
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
                                // Dont sleep more than 1 sec
                                if(sleepTime > 1000000) {
                                    sleepTime = 1000000;
                                    qCritical("Sleep time more than 1 sec. Truncated!");
                                }
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
        // Process incoming events
        QCoreApplication::processEvents();
    }
}

void eDVSInterface::stopWork()
{
    operationMutex.lock();
    OperationMode localOpMode = operationMode;
    operationMutex.unlock();

    // Stop streaming
    if(localOpMode == STREAMING)
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

    // read file, parse header
    EventBuilder::TimestampVersion timeVers;
    EventBuilder::AddressVersion addrVers;
    QByteArray bytes = parseEventFile(fileName,addrVers,timeVers);
    // Init eventbuilder
    evBuilder.initEvBuilder(addrVers,timeVers);

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
        if(evBuilder.evBuilderProcessNextByte(bytes.at(bufferIdx++),eNew,false)) {
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
                    if(sleepTime > 1000000)
                        qDebug("[eDVSInterface] Warning! Sleeping %u us until next event!",sleepTime);
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
#ifdef QT_DEBUG
        quint32 elapsedTimeReal = timeMeasure.nsecsElapsed()/1000;
        quint32 elapsedTimeEvents = eNew.timestamp - startTimestamp;
        PRINT_DEBUG_FMT("%s", QString("Executed %1 events in %2 ms instead of %3 ms. Overhead: %4 %")
                        .arg(eventCount)
                        .arg(elapsedTimeReal/1000.0)
                        .arg(elapsedTimeEvents/speed/1000.0)
                        .arg((static_cast<double>(elapsedTimeReal)/(elapsedTimeEvents/speed) - 1) *100)
                        .toLocal8Bit().data());
#endif

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
}

QByteArray eDVSInterface::parseEventFile(QString file, EventBuilder::AddressVersion &addrVers, EventBuilder::TimestampVersion &timeVers)
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
    addrVers = EventBuilder::Addr2Byte;
    if(versionNr == 2) {
        addrVers = EventBuilder::Addr4Byte;
    }
    timeVers = EventBuilder::Time4Byte;

#ifdef QT_DEBUG
    size_t eventCnt = buff.size()/(addrVers+4);
    PRINT_DEBUG_FMT("%s", QString("%1 Events.").arg(eventCnt).toLocal8Bit().data());
#endif

    return buff;
}
