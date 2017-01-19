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
    qDebug("Destroying eDVSInterface...");

    operationMutex.lock();
    operationMode = IDLE;
    operationMutex.unlock();

    thread.quit();

    if(!thread.wait(2000)){
        thread.terminate();
        thread.wait();
    }

    if(evBuilderData != NULL)
        delete[] evBuilderData;
    evBuilderData = NULL;
}

void eDVSInterface::setPushBotCtrl(PushBotController* pushBotCtrl){
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
        socket.write("E1\n");
        socket.write("E+\n");
        socket.waitForBytesWritten();
    }
    initEvBuilder(Addr2Byte,TimeDelta);
    {
        QMutexLocker locker(&operationMutex);
        operationMode = ONLINE_STREAMING;
    }
    processingWorker->startProcessing();
}
void eDVSInterface::stopEventStreaming()
{
    {
        QMutexLocker locker(&operationMutex);
        operationMode = ONLINE;
    }
    {
        QMutexLocker locker(&socketMutex);
        socket.write("E-\n");
        socket.waitForBytesWritten();
    }
    processingWorker->stopProcessing();
}
void eDVSInterface::sendRawCmd(QString cmd)
{
    OperationMode opModeLocal;
    {
        QMutexLocker locker(&operationMutex);
        opModeLocal = operationMode;
    }

    QMutexLocker locker(&socketMutex);
    socket.write(cmd.toLocal8Bit());
    socket.waitForBytesWritten();
    emit onCmdSent(cmd);
}
void eDVSInterface::enableMotors(bool enable){

    OperationMode opModeLocal;
    {
        QMutexLocker locker(&operationMutex);
        opModeLocal = operationMode;
    }

    if(opModeLocal == ONLINE){
        QMutexLocker locker(&socketMutex);
        if(enable)
            socket.write("M+\n");
        else
            socket.write("M-\n");
        socket.waitForBytesWritten();
    }
}
void eDVSInterface::setMotorVelocity(int motorId, int speed)
{
    OperationMode opModeLocal;
    {
        QMutexLocker locker(&operationMutex);
        opModeLocal = operationMode;
    }

    if(opModeLocal == ONLINE){
        QMutexLocker locker(&socketMutex);
        socket.write(QString("MV%1=%2\n").arg(motorId).arg(speed).toLocal8Bit());
        socket.waitForBytesWritten();
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

    // Wait for processing stopped
    thread.quit();

    qDebug("Done");
}
void eDVSInterface::_processSocket(){
    // Init serial port
    socket.connectToHost(host,port);
    qDebug("Connecting to %s: %d",host.toLocal8Bit().data(),port);
    if(!socket.waitForConnected(2000)){
        operationMutex.lock();
        operationMode = IDLE;
        operationMutex.unlock();
        emit onConnectionResult(true);
        qDebug("Can't connect to socket \"%s:%d\": %s"
               ,host.toLocal8Bit().data(),port,socket.errorString().toLocal8Bit().data());
        return;
    }
    emit onConnectionResult(false);
    qDebug("Connection established");
    OperationMode opModeLocal;
    {
        QMutexLocker locker2(&operationMutex);
        opModeLocal = operationMode;
    }

    DVSEvent eNew;
    while(opModeLocal == ONLINE || opModeLocal == ONLINE_STREAMING){
        {
            QMutexLocker locker(&socketMutex);
            if(socket.state() != QTcpSocket::ConnectedState){
                qDebug("Connection closed: %s",socket.errorString().toLocal8Bit().data());
                QMutexLocker locker2(&operationMutex);
                operationMode = IDLE;
                emit onConnectionClosed(true);
                break;
            }

            // Normal command mode
            if(opModeLocal == ONLINE){
                if(socket.waitForReadyRead(10)){
                    QString line = QString(socket.readAll()).remove(QRegExp("[\\n\\t\\r]"));;
                    qDebug("Recieved: %s",line.toLocal8Bit().data());
                    emit onLineRecived(line);
                }
            // Streaming mode
            }else{
                if(socket.bytesAvailable())
                {
                    char c;
                    socket.getChar(&c);
                    if(processingWorker != NULL &&
                            evBuilderProcessNextByte(c,eNew)){
                        processingWorker->nextEvent(eNew);
                    }
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

    qDebug("Stopping playback/stream/communication...");
    if(!thread.wait(2000)){
        thread.terminate();
        thread.wait();
        qDebug("Stopped.");
    }
    // Close socket
    if(localOpMode == ONLINE)
    {
        QMutexLocker locker2(&socketMutex);
        socket.disconnectFromHost();
        emit onConnectionClosed(false);
    }
}

void eDVSInterface::playbackFile(QString fileName, float speed)
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
    float speed;
    QString fileName;
    {
        QMutexLocker locker2(&playbackDataMutex);
        speed = playbackSpeed;
        fileName = playbackFileName;
    }

    Worker* workerLocal;
    PushBotController* pushBotControllerLocal;

    {
        QMutexLocker locker(&operationMutex);
        workerLocal = processingWorker;
        pushBotControllerLocal = pushBotController;
    }

    if(workerLocal == NULL)
        return;

    qDebug("Start worker");
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

    long bufferIdx = 0;
    DVSEvent eNew;
    long startTimestamp = -1;
    long eventCount = 0;
    // Measure real time
    timeMeasure.start();
    do{
        // New event ready ?
        if(evBuilderProcessNextByte(bytes.at(bufferIdx++),eNew)){
//            if(eNew.On)
//                continue;
            eventCount++;

            // send first event directly
            if(startTimestamp == -1){
                processingWorker->nextEvent(eNew);
                startTimestamp = eNew.timestamp;
            }
            // Compute sleep time
            else{
                long elapsedTimeReal = timeMeasure.nsecsElapsed()/1000;
                long elapsedTimeEvents = (eNew.timestamp - startTimestamp)/speed;
                long sleepTime = elapsedTimeEvents - elapsedTimeReal;

                //qDebug("%ld %ld %ld",elapsedTimeReal,elapsedTimeEvents,sleepTime);

                // Sleep if necessary
                sleepTime = qMax(0L,sleepTime);
                if(sleepTime > 0){
                    //qDebug("Sleep: %ld",sleepTime);
                    QThread::usleep(sleepTime);
                }
                //qDebug("Time: %ld",eNew.timestamp);
                processingWorker->nextEvent(eNew);
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

        {
            QMutexLocker locker2(&operationMutex);
            operationMode = IDLE;
        }
        emit onPlaybackFinished();
    }


    {
        QMutexLocker locker(&operationMutex);
        workerLocal = processingWorker;
        pushBotControllerLocal = pushBotController;
    }
    qDebug("Stop worker");
    workerLocal->stopProcessing();
    emit onStopPushBotController();
}

QByteArray eDVSInterface::parseEventFile(QString file, AddressVersion &addrVers, TimestampVersion &timeVers)
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

void eDVSInterface::initEvBuilder(AddressVersion addrVers, TimestampVersion timeVers)
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

bool eDVSInterface::evBuilderProcessNextByte(char c, DVSEvent &event)
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

eDVSInterface::DVSEvent eDVSInterface::evBuilderParseEvent()
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
