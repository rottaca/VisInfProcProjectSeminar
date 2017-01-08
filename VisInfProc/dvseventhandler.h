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

    DVSEventHandler(QObject* parent = 0);
    ~DVSEventHandler();

    void setWorker(Worker* worker){
        operationMutex.lock();
        this->worker = worker;
        operationMutex.unlock();
    }

    void abort();

    void run();

signals:
    void OnPlaybackFinished();

public slots:
    void playbackFile(QString fileName, float speed);


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
};
#endif // DVSEVENTHANDLER_H
