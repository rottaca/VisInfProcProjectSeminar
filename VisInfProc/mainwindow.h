#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QTime>

#include "buffer1d.h"
#include "buffer2d.h"
#include "buffer3d.h"
#include "filtermanager.h"
#include "filtersettings.h"
#include "filterset.h"
#include "opticflowestimator.h"
#include "worker.h"
#include "edvsinterface.h"
#include "settings.h"
#include "pushbotcontroller.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

public slots:
    void onConnectionResult(bool failed);
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
    QTimer updateTimer;
    QTime lastStatisticsUpdate;
    QVector<float> orientations;
    QVector<FilterSettings> settings;
    Buffer2D oppMoEnergy1,oppMoEnergy2, flowX,flowY;


    Worker worker;
    eDVSInterface eDVSHandler;
    PushBotController pushBotController;
};

#endif // MAINWINDOW_H
