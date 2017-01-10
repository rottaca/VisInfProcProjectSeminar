#ifndef SERIALEDVSINTERFACE_H
#define SERIALEDVSINTERFACE_H

#include <QSerialPort>
#include <QThread>
#include <QMutex>
#include <QMutexLocker>


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


    // Connect/Disconnect
    bool open(QString portName);
    void stop();


    // Commands
    void startEventStreaming();
    void stopEventStreaming();
    void enableMotors(bool enable);
    void setMotorVelocity(int motorId, int speed);

    // Playback
    void playbackFile(QString fileName, float speed);

    // Getters / Setters
    bool isConnected(){
        QMutexLocker locker(&serialMutex);
        return serial.isOpen();
    }

    bool isStreaming(){
        QMutexLocker locker(&operationMutex);
        return operationMode == ONLINE_STREAMING;
    }

    void setWorker(Worker* worker){
        QMutexLocker locker(&operationMutex);
        this->worker = worker;
    }

signals:
    void onPlaybackFinished();
    void onLineRecived(QString answ);
    void onCmdSent(QString cmd);


public slots:
    void process();
    void sendRawCmd(QString cmd);


private:
    // Playback function to parse and play an event file
    QByteArray parseEventFile(QString file, AddressVersion &addrVers, TimestampVersion &timeVers);
    void _playbackFile();

    // Event builder functions
    void initEvBuilder(AddressVersion addrVers, TimestampVersion timeVers);
    bool evBuilderProcessNextByte(char c, DVSEvent &event);
    DVSEvent evBuilderParseEvent();

private:
    // Thread for async processing of playback file and serial interface
    QThread thread;
    // Operation mode and worker thread
    typedef enum OperationMode{IDLE,PLAYBACK,ONLINE,ONLINE_STREAMING} OperationMode;
    OperationMode operationMode;
    Worker *worker;
    QMutex operationMutex;

    // Serial port for realtime processing
    QSerialPort serial;
    QMutex serialMutex;

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
