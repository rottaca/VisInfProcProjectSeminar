#ifndef DVSEVENTHANDLER_H
#define DVSEVENTHANDLER_H
#include <QTimer>
#include <QVector>




class DVSEventHandler: public QObject
{
    Q_OBJECT
public:

    typedef struct {
        int posX, posY;
        bool On;
        long timestamp;
    } DVSEvent;

    DVSEventHandler(QObject* parent = 0);

    bool playBackFile(QString fileName, int speedMs);


signals:
    void OnNewEvent(DVSEventHandler::DVSEvent e);

public slots:
    void onTimePlayback();

private:
    QTimer timer;
    QVector<DVSEvent> eventList;
    int eventIdx;
};

Q_DECLARE_METATYPE(DVSEventHandler::DVSEvent)

#endif // DVSEVENTHANDLER_H
