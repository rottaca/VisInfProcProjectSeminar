#ifndef SERIALEDVSINTERFACE_H
#define SERIALEDVSINTERFACE_H

#include <QSerialPort>
#include <QThread>
#include <QMutex>
#include <QMutexLocker>

#include "dvseventhandler.h"

class SerialeDVSInterface: public QObject
{
    Q_OBJECT
public:
    SerialeDVSInterface(QObject* parent = 0);

    // Connect/Disconnect
    bool open(QString portName);
    void close();



    // Getters
    bool isConnected(){return opened;}
    bool isStreaming(){return streaming;}



    // Commands
    void startEventStreaming();
    void stopEventStreaming();
    void enableMotors(bool enable);
    void setMotorVelocity(int motorId, int speed);




    void setDVSEventHandler(DVSEventHandler* eventHandler){
        QMutexLocker locker(&dataMutex);
        this->eventHandler = eventHandler;
    }

signals:
    void onLineRecived(QString answ);
    void onCmdSent(QString cmd);

public slots:
    void run();
    void sendRawCmd(QString cmd);

private:
    QThread thread;
    bool opened;
    QSerialPort serial;
    QMutex serialMutex;
    bool streaming;
    DVSEventHandler* eventHandler;
    QMutex dataMutex;
};

#endif // SERIALEDVSINTERFACE_H
