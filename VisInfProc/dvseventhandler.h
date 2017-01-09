#ifndef DVSEVENTHANDLER_H
#define DVSEVENTHANDLER_H
#include <QTimer>
#include <QVector>
#include <QTime>
#include <QThread>
#include <QMutex>

// Forward declaration
class Worker;

class DVSEventHandler: public QThread
{
    Q_OBJECT
public:

    typedef struct {
        u_int8_t posX, posY;
        u_int8_t On;
        u_int32_t timestamp;
    } DVSEvent;

    typedef enum AddressVersion{Addr2Byte,Addr4Byte} AddressVersion;
    typedef enum TimestampVersion{Time4Byte, Time3Byte, Time2Byte, TimeDelta, TimeNoTime} TimestampVersion;

    DVSEventHandler(QObject* parent = 0);
    ~DVSEventHandler();

    void setWorker(Worker* worker){
        operationMutex.lock();
        this->worker = worker;
        operationMutex.unlock();
    }

    DVSEvent parseEvent(uchar *data, int sz, AddressVersion addrVers, TimestampVersion timeVers);

    void nextRealtimeEventByte(char c);

    void abort();

    void run();

signals:
    void onPlaybackFinished();

public slots:
    void playbackFile(QString fileName, float speed);
    void initStreaming(TimestampVersion timeVers);

private:
    QVector<DVSEvent> parseFile(QByteArray &buff);
    void _playbackFile();

private:
    QString playbackFileName;
    float playbackSpeed;
    QMutex playbackDataMutex;

    typedef enum OperationMode{IDLE,PLAYBACK,ONLINE} OperationMode;
    OperationMode operationMode;
    Worker *worker;
    QMutex operationMutex;

    QMutex realtimeDataMutex;
    TimestampVersion realtimeTimestampVersion;
    int realtimeEventByteIdx;
    int realtimeEventBufferSize;
    uchar * realtimeEventData;
    long realtimeEventTimestamp;

};
#endif // DVSEVENTHANDLER_H
