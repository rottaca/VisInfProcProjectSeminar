#ifndef SERIALEDVSINTERFACE_H
#define SERIALEDVSINTERFACE_H

#include <QThread>
#include <QMutex>
#include <QMutexLocker>
#include <QTcpSocket>

class Worker;

class SerialeDVSInterface: public QObject
{
    Q_OBJECT
public:

    typedef struct DVSEvent{
        u_int8_t posX, posY;
        u_int8_t On;
        u_int32_t timestamp;
    } DVSEvent;

    typedef enum AddressVersion{Addr2Byte = 2,Addr4Byte = 4} AddressVersion;
    typedef enum TimestampVersion{Time4Byte = 4, Time3Byte = 3, Time2Byte = 2, TimeDelta = -1, TimeNoTime = 0} TimestampVersion;

    SerialeDVSInterface(QObject* parent = 0);
    ~SerialeDVSInterface();

    // Getters / Setters
    bool isConnected(){
        QMutexLocker locker(&operationMutex);
        return operationMode == ONLINE_STREAMING || operationMode == ONLINE;
    }

    bool isStreaming(){
        QMutexLocker locker(&operationMutex);
        return operationMode == ONLINE_STREAMING;
    }

    bool isWorking(){
        QMutexLocker locker(&operationMutex);
        return operationMode != IDLE;
    }



    void setWorker(Worker* worker){
        QMutexLocker locker(&operationMutex);
        this->processingWorker = worker;
    }

    // Connect/Disconnect/Playback
    void playbackFile(QString fileName, float speed);
    void connectToBot(QString host, int port);
    void stopWork();

    // Commands
    void sendRawCmd(QString cmd);
    void startEventStreaming();
    void stopEventStreaming();
    void enableMotors(bool enable);
    void setMotorVelocity(int motorId, int speed);

signals:
    void onPlaybackFinished();
    void onConnectionResult(bool failed);
    void onLineRecived(QString answ);
    void onCmdSent(QString cmd);

public slots:

    void process();


private:
    // Playback function to parse and play an event file
    QByteArray parseEventFile(QString file, AddressVersion &addrVers, TimestampVersion &timeVers);
    void _playbackFile();

    void _processSocket();

    // Event builder functions
    void initEvBuilder(AddressVersion addrVers, TimestampVersion timeVers);
    bool evBuilderProcessNextByte(char c, DVSEvent &event);
    DVSEvent evBuilderParseEvent();

private:
    // Thread for async processing of playback file and tcp socket
    QThread thread;
    // Operation mode and worker thread
    typedef enum OperationMode{IDLE,PLAYBACK,ONLINE,ONLINE_STREAMING} OperationMode;
    OperationMode operationMode;
    Worker *processingWorker;
    QMutex operationMutex;

    // Tcp connection for realtime processing
    QTcpSocket socket;
    QString host;
    int port;
    QMutex socketMutex;

    // Playback data
    QString playbackFileName;
    float playbackSpeed;
    QMutex playbackDataMutex;

    // Data for the event builder
    TimestampVersion evBuilderTimestampVersion;
    AddressVersion evBuilderAddressVersion;
    int evBuilderByteIdx;
    int evBuilderBufferSz;
    char* evBuilderData;
    long evBuilderSyncTimestamp;
    QMutex evBuilderMutex;

};

#endif // SERIALEDVSINTERFACE_H
