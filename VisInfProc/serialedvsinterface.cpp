#include "serialedvsinterface.h"
#include <QApplication>

SerialeDVSInterface::SerialeDVSInterface(QObject *parent):QObject(parent)
{
    moveToThread(&thread);
    serial.moveToThread(&thread);

    connect(&thread,SIGNAL(started()),this,SLOT(run()));


    opened = false;
    streaming = false;
    eventHandler = NULL;
}

bool SerialeDVSInterface::open(QString portName)
{
    QMutexLocker locker(&dataMutex);
    QMutexLocker locker2(&serialMutex);

    // Init serial port
    serial.setBaudRate(4000000);
    serial.setStopBits(QSerialPort::OneStop);
    serial.setDataBits(QSerialPort::Data8);
    serial.setParity(QSerialPort::NoParity);
    serial.setFlowControl(QSerialPort::HardwareControl);
    serial.setPortName(portName);

    if(serial.error() != QSerialPort::NoError){
        qDebug("Serial port error: %d",serial.error());
        return false;
    }

    if(!serial.open(QSerialPort::ReadWrite))
        return false;

    opened = true;

    thread.start();
    return true;
}
void SerialeDVSInterface::close()
{
    if(isStreaming())
        stopEventStreaming();

    dataMutex.lock();
    opened = false;
    streaming = false;
    dataMutex.unlock();

    if(!thread.wait(2000)){
        thread.terminate();
        thread.wait();
    }
}

void SerialeDVSInterface::startEventStreaming()
{
    // TODO
    QMutexLocker locker2(&dataMutex);
    if(eventHandler != NULL)
        eventHandler->initStreaming(DVSEventHandler::TimeDelta);

    QMutexLocker locker(&serialMutex);
    serial.write("E1\n");
    serial.write("E+\n");
    serial.waitForBytesWritten(10);

    streaming = true;
}
void SerialeDVSInterface::stopEventStreaming()
{
    // TODO
    streaming = false;
    QMutexLocker locker(&serialMutex);
    serial.write("E-\n");
    serial.waitForBytesWritten(10);
    QMutexLocker locker2(&dataMutex);
    if(eventHandler != NULL)
        eventHandler->abort();
}
void SerialeDVSInterface::sendRawCmd(QString cmd)
{
    dataMutex.lock();
    bool openedLocal = opened;
    dataMutex.unlock();

    if(openedLocal){
        QMutexLocker locker(&serialMutex);
        serial.write(cmd.toLocal8Bit());
        emit onCmdSent(cmd);
    }
}
void SerialeDVSInterface::enableMotors(bool enable){

    dataMutex.lock();
    bool openedLocal = opened;
    dataMutex.unlock();

    if(openedLocal){
        QMutexLocker locker(&serialMutex);
        if(enable)
            serial.write("M+\n");
        else
            serial.write("M-\n");
        serial.waitForBytesWritten(10);
    }
}
void SerialeDVSInterface::setMotorVelocity(int motorId, int speed)
{
    dataMutex.lock();
    bool openedLocal = opened;
    dataMutex.unlock();

    if(openedLocal){
        QMutexLocker locker(&serialMutex);
        serial.write(QString("MV%1=%2\n").arg(motorId).arg(speed).toLocal8Bit());
        serial.waitForBytesWritten(10);
    }
}

void SerialeDVSInterface::run()
{
    dataMutex.lock();
    bool openedLocal = opened;
    bool streamingLocal = streaming;
    dataMutex.unlock();

    while(openedLocal){
        {
            QMutexLocker locker(&serialMutex);
            // Normal command mode
            if(!streamingLocal){
                if(serial.waitForReadyRead(10) &&
                        serial.canReadLine()){
                    QString line = serial.readLine();
                    emit onLineRecived(line);
                }
            // Streaming mode
            }else{
                if(serial.waitForReadyRead(10))
                {
                    char c;
                    serial.getChar(&c);
                    // Event read -> deliver
                    if(eventHandler != NULL){
                        eventHandler->nextRealtimeEventByte(c);
                    }
                }
            }
        }
        dataMutex.lock();
        openedLocal = opened;
        streamingLocal = streaming;
        dataMutex.unlock();

        qApp->processEvents();
    }
}
