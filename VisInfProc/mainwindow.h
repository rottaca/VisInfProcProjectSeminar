#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QTime>
#include <QImage>
#include <QElapsedTimer>

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

class FilterSelectionForm;

namespace Ui
{
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();


    void closeEvent(QCloseEvent *bar);

public slots:
    void onConnectionResult(bool error);
    void onUpdate();
    void onPlaybackFinished();
    void onClickStartPlayback();
    void onChangePlaybackFile();
    void onLineRecived(QString answ);
    void onCmdSent(QString cmd);
    void onClickStartStreaming();
    void onClickConnect();
    void onCmdEntered();
    void onConnectionClosed(bool error);
    void onClickReset();
    void onChangeThreshold(double v);
    void onChangePushbotP(double v);
    void onChangePushbotI(double v);
    void onChangePushbotD(double v);
    void onChangeRenderMode();
    void activeFiltersChanged(QVector<int> activeOrientationIndices,
                              QVector<int> activeSettingIndices);
    void onClickChangeActiveFilters();
    void onClickStartNavigation();
    void onStreamingStarted();
    void onStreamingStopped();
    void stopAll();

signals:
    void playbackFile(QString fileName, double speed);
    //void connectToBot(QString host, int port);
    void connectToBot(QString port);
    void sendRawCmd(QString cmd);
    void startEventStreaming();
    void stopEventStreaming();
    void eDVSInterfaceStopWork();
    void reset();
    void startPushBotController();
    void stopPushBotController();

private:
    void initUI();
    void initSystem();
    void initSignalsAndSlots();
    void setupActiveFilters();

private:
    cudaStream_t cudaStream;
    Ui::MainWindow *ui;
    FilterSelectionForm* filterSelectionForm;
    QTimer updateTimer;
    QTime lastStatisticsUpdate;

    QElapsedTimer timer;

    QVector<float> allOrientations;
    QVector<FilterSettings> allSettings;
    QVector<int> activeOrientationIndices;
    QVector<int> activeSettingIndices;

    Buffer2D motionEnergy, speed,energy,dir;
    QImage rgbImg;
    char* gpuRgbImage;

    Worker worker;
    eDVSInterface eDVSHandler;
    PushBotController pushBotController;
};

#endif // MAINWINDOW_H
