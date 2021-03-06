#ifndef SERIALEDVSINTERFACE_H
#define SERIALEDVSINTERFACE_H

#include <QThread>
#include <QMutex>
#include <QMutexLocker>
#include <QTcpSocket>

#include "datatypes.h"
#include "eventbuilder.h"

#include <QtSerialPort/QtSerialPort>

class Worker;
class PushBotController;

class eDVSInterface: public QObject
{
    Q_OBJECT
public:
    eDVSInterface(QObject* parent = 0);
    ~eDVSInterface();


signals:
    /**
     * @brief onPlaybackFinished Emited when the playback has finished
     */
    void onPlaybackFinished();
    /**
     * @brief onConnectionClosed Emited when the connecion is closed
     */
    void onConnectionClosed(bool error);
    /**
     * @brief onConnectionResult Emitted after trying to connect to the robot.
     * @param failed True if failed.
     */
    void onConnectionResult(bool failed);
    /**
     * @brief onLineRecived Emitted when a the robot sends a resonse.
     * @param answ
     */
    void onLineRecived(QString answ);
    /**
     * @brief onCmdSent Emitted when a command was send
     * @param cmd
     */
    void onCmdSent(QString cmd);

    void onStreamingStarted();

    void onStreamingStopped();


public slots:
    /**
     * @brief playbackFile Plays a given file with a given speed. Default = 1.0
     * @param fileName
     * @param speed
     */
    void playbackFile(QString fileName, double speed = 1.0f);
    /**
     * @brief connectToBot Connects to the robot platform.
     * @param host
     * @param port
     */
    //void connectToBot(QString host, int port);
    void connectToBot(QString port);
    /**
     * @brief sendRawCmd Sends the provided command to the robot if connected
     * @param cmd
     */
    void sendRawCmd(QString cmd);
    /**
     * @brief startEventStreaming If connected, this functions sends the commands to start the event streaming
     */
    void startEventStreaming();
    /**
     * @brief stopEventStreaming Stops the event streaming
     */
    void stopEventStreaming();
    /**
     * @brief enableMotors Sends cmds to start or stop the motors
     * @param enable
     */
    void enableMotors(bool enable);
    /**
     * @brief setMotorVelocity Sets the motor velocity for a given motor
     * @param motorId
     * @param speed
     */
    void setMotorVelocity(int motorId, int speed);

    void resetBoard();

    /**
     * @brief stopWork Stops playback of file or closes connection to robot.
     */
    void stopWork();

public:
    // Getters / Setters
    /**
     * @brief isConnected Returns true, when the system is connected to the robot
     * @return
     */
    bool isConnected()
    {
        QMutexLocker locker(&operationMutex);
        return operationMode != IDLE && operationMode != PLAYBACK;
    }
    /**
     * @brief isStreaming Returns true when online event streaming from the robot is enabled
     * @return
     */
    bool isStreaming()
    {
        QMutexLocker locker(&operationMutex);
        return operationMode == STREAMING;
    }
    /**
     * @brief isWorking Returns true, when the eDVS interface is online or in playback mode
     * @return
     */
    bool isWorking()
    {
        QMutexLocker locker(&operationMutex);
        return operationMode != IDLE;
    }
    /**
     * @brief setWorker Set the pointer to the async processor
     * @param worker
     */
    void setWorker(Worker* worker)
    {
        QMutexLocker locker(&operationMutex);
        processingWorker = worker;
    }
    /**
     * @brief setPushBotCtrl Set the pointer to the async processor
     * @param pushBotCtrl
     */
    void setPushBotCtrl(PushBotController* pushBotCtrl);

private:
    // Playback function to parse and play an event file
    QByteArray parseEventFile(QString file, EventBuilder::AddressVersion &addrVers, EventBuilder::TimestampVersion &timeVers);
    /**
     * @brief _playbackFile Loads and plays an event file
     */
    void _playbackFile();
    /**
     * @brief _processSocket Sends and recieves commands from and to the eDVS.
     */
    void _processSocket(QString port);

private:
    // Thread for async processing of playback file and tcp socket
    QThread thread;
    // Operation mode
    typedef enum OperationMode {IDLE,PLAYBACK,ONLINE,START_STREAMING,STREAMING, STOP_STREAMING} OperationMode;
    OperationMode operationMode;
    // Pointer to processing thread
    Worker *processingWorker;
    PushBotController *pushBotController;
    QMutex operationMutex;

    // Tcp connection for realtime processing
    //QTcpSocket socket;
    QSerialPort serialPort;
    //QString host;
    //int port;
    QMutex socketMutex;

    // Playback data
    QString playbackFileName;
    double playbackSpeed;
    QMutex playbackDataMutex;

    EventBuilder evBuilder;
};

#endif // SERIALEDVSINTERFACE_H
