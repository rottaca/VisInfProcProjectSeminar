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

    void run();

signals:
    void OnPlaybackFinished();

public slots:
    void PlayBackFile(QString fileName, int speedus = -1);

private:
    QVector<DVSEvent> parseFile(QByteArray &buff);
    void playbackFile();

private:
    QString playbackFileName;
    int playbackSpeed;
    QMutex playbackDataMutex;

    typedef enum OperationMode{IDLE,PLAYBACK,ONLINE} OperationMode;
    OperationMode operationMode;
    Worker *worker;
    QMutex operationMutex;
};
#endif // DVSEVENTHANDLER_H
