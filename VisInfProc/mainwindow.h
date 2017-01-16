#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>

#include "buffer1d.h"
#include "buffer2d.h"
#include "buffer3d.h"
#include "filtermanager.h"
#include "filtersettings.h"
#include "filterset.h"
#include "convolution3d.h"
#include "dvseventhandler.h"
#include "opticflowestimator.h"
#include "worker.h"
#include "serialedvsinterface.h"

#define FPS 25


namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

signals:
    void startProcessing();
    void sendRawCmd(QString cmd);

public slots:
    void onUpdate();
    void onPlaybackFinished();
    void onClickStartPlayback();
    void onChangePlaybackFile();
    void onLineRecived(QString answ);
    void onCmdSent(QString cmd);
    void onClickStartStreaming();
    void onClickConnect();
    void onCmdEntered();

private:
    void initUI();
    void initSystem();
    void initSignalsAndSlots();

private:
    cudaStream_t cudaStream;
    Ui::MainWindow *ui;
    FilterSettings fsettings;
    DVSEventHandler dvsEventHandler;
    Worker* worker;
    QTimer updateTimer;
    QTime lastStatisticsUpdate;
    QVector<float> orientations;
    QVector<FilterSettings> settings;
    SerialeDVSInterface serial;
    Buffer2D oppMoEnergy1,oppMoEnergy2, flowX,flowY;
};

#endif // MAINWINDOW_H
