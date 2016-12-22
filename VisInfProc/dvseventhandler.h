#ifndef DVSEVENTHANDLER_H
#define DVSEVENTHANDLER_H
#include <QTimer>
#include <QVector>




class DVSEventHandler: public QObject
{
    Q_OBJECT
public:

    typedef struct {
        u_int8_t posX, posY;
        u_int8_t On;
        u_int32_t timestamp;
    } DVSEvent;

    DVSEventHandler(QObject* parent = 0);

    bool PlayBackFile(QString fileName, int speedMs);


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
