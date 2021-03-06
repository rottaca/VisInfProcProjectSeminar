#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "cuda_helper.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <nvToolsExt.h>

#include <QtMath>
#include <QList>
#include <QPainter>
#include <QPoint>
#include <QLine>
#include <QtMath>
#include <QFileDialog>
#include <QtSerialPort/QSerialPortInfo>
#include <QtSerialPort/QSerialPort>
#include <QMessageBox>

#include "settings.h"
#include "filterselectionform.h"

bool sortSettingsBySpeed(const FilterSettings &a, const FilterSettings &b)
{
    return a.speed_px_per_sec < b.speed_px_per_sec;
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    initSystem();
    initUI();
    initSignalsAndSlots();

    updateTimer.start(1000/GUI_RENDERING_FPS);
}
MainWindow::~MainWindow()
{
    gpuErrchk(cudaFree(gpuRgbImage));
    delete ui;
    cudaStreamDestroy(cudaStream);
}

void MainWindow::initUI()
{
    ui->setupUi(this);

    filterSelectionForm = new FilterSelectionForm();
    lastStatisticsUpdate.start();


    rgbImg = QImage(DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_HEIGHT,QImage::Format_RGB888);
    gpuErrchk(cudaMalloc(&gpuRgbImage,DVS_RESOLUTION_WIDTH*DVS_RESOLUTION_HEIGHT*3));

    ui->sb_energy_threshold->setValue(FLOW_DEFAULT_MIN_ENERGY_THRESHOLD);
    ui->sb_pushbot_p->setValue(PUSHBOT_PID_P_DEFAULT);
    ui->sb_pushbot_i->setValue(PUSHBOT_PID_I_DEFAULT);
    ui->sb_pushbot_d->setValue(PUSHBOT_PID_D_DEFAULT);

    setupActiveFilters();

    onChangeRenderMode();
}

void MainWindow::initSystem()
{
    gpuErrchk(cudaSetDevice(0));
    cudaStreamCreate(&cudaStream);

    energy.setCudaStream(cudaStream);
    speed.setCudaStream(cudaStream);
    dir.setCudaStream(cudaStream);
    motionEnergy.setCudaStream(cudaStream);
    energy.resize(DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_WIDTH);
    speed.resize(DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_WIDTH);
    dir.resize(DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_WIDTH);
    motionEnergy.resize(DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_WIDTH);

    // Add all avaiable speeds
    allSettings.append(FilterSettings::getSettings(FilterSettings::SPEED_1));
    allSettings.append(FilterSettings::getSettings(FilterSettings::SPEED_2));
    allSettings.append(FilterSettings::getSettings(FilterSettings::SPEED_3));
    allSettings.append(FilterSettings::getSettings(FilterSettings::SPEED_4));
    allSettings.append(FilterSettings::getSettings(FilterSettings::SPEED_5));

    // Sort list of settings by speed for interpolation
    std::sort(allSettings.begin(),allSettings.end(),sortSettingsBySpeed);

    // Add all available orientations
    allOrientations.append(DEG2RAD(0.0f));
    allOrientations.append(DEG2RAD(180.0f));
    allOrientations.append(DEG2RAD(90.0f));
    allOrientations.append(DEG2RAD(-90.0f));
    allOrientations.append(DEG2RAD(45.0f));
    allOrientations.append(DEG2RAD(-45.0f));
    allOrientations.append(DEG2RAD(135.0f));
    allOrientations.append(DEG2RAD(-135.0f));

    // Set default speeds and orientations
    activeOrientationIndices = {0,1,2,3};
    activeSettingIndices = {0,1,2};

    eDVSHandler.setWorker(&worker);
    eDVSHandler.setPushBotCtrl(&pushBotController);
    pushBotController.setWorker(&worker);
    pushBotController.setRobotInterface(&eDVSHandler);
}

void MainWindow::initSignalsAndSlots()
{
    connect(&updateTimer,SIGNAL(timeout()),this,SLOT(onUpdate()));

    connect(this,SIGNAL(sendRawCmd(QString)),&eDVSHandler,SLOT(sendRawCmd(QString)));
    connect(this,SIGNAL(startEventStreaming()),&eDVSHandler,SLOT(startEventStreaming()));
    connect(this,SIGNAL(stopEventStreaming()),&eDVSHandler,SLOT(stopEventStreaming()));
    connect(this,SIGNAL(reset()),&eDVSHandler,SLOT(resetBoard()));
    connect(this,SIGNAL(eDVSInterfaceStopWork()),&eDVSHandler,SLOT(stopWork()));
    connect(this,SIGNAL(eDVSInterfaceStopWork()),&eDVSHandler,SLOT(stopWork()));
    connect(this,SIGNAL( playbackFile(QString, double)),&eDVSHandler,SLOT( playbackFile(QString, double)));
    //connect(this,SIGNAL(connectToBot(QString, int)),&eDVSHandler,SLOT(connectToBot(QString, int)));
    connect(this,SIGNAL(connectToBot(QString)),&eDVSHandler,SLOT(connectToBot(QString)));

    connect(&eDVSHandler,SIGNAL(onCmdSent(QString)),this,SLOT(onCmdSent(QString)));
    connect(&eDVSHandler,SIGNAL(onLineRecived(QString)),this,SLOT(onLineRecived(QString)));
    connect(&eDVSHandler,SIGNAL(onPlaybackFinished()),this,SLOT(onPlaybackFinished()));
    connect(&eDVSHandler,SIGNAL(onConnectionResult(bool)),this,SLOT(onConnectionResult(bool)));
    connect(&eDVSHandler,SIGNAL(onConnectionClosed(bool)),this,SLOT(onConnectionClosed(bool)));
    connect(&eDVSHandler,SIGNAL(onStreamingStarted()),this,SLOT(onStreamingStarted()));
    connect(&eDVSHandler,SIGNAL(onStreamingStopped()),this,SLOT(onStreamingStopped()));

    connect(ui->b_browse_play_file,SIGNAL(clicked()),this,SLOT(onChangePlaybackFile()));
    connect(ui->b_start_playback,SIGNAL(clicked()),this,SLOT(onClickStartPlayback()));
    connect(ui->b_connect,SIGNAL(clicked()),this,SLOT(onClickConnect()));
    connect(ui->b_start_streaming,SIGNAL(clicked()),this,SLOT(onClickStartStreaming()));
    connect(ui->le_cmd_input,SIGNAL(returnPressed()),this,SLOT(onCmdEntered()));
    connect(ui->b_reset,SIGNAL(clicked()),this,SLOT(onClickReset()));
    connect(ui->sb_energy_threshold,SIGNAL(valueChanged(double)),this,SLOT(onChangeThreshold(double)));
    connect(ui->sb_pushbot_p,SIGNAL(valueChanged(double)),this,SLOT(onChangePushbotP(double)));
    connect(ui->sb_pushbot_i,SIGNAL(valueChanged(double)),this,SLOT(onChangePushbotI(double)));
    connect(ui->sb_pushbot_d,SIGNAL(valueChanged(double)),this,SLOT(onChangePushbotD(double)));
    connect(ui->rb_debug,SIGNAL(clicked()),this,SLOT(onChangeRenderMode()));
    connect(ui->rb_normal,SIGNAL(clicked()),this,SLOT(onChangeRenderMode()));
    connect(ui->rb_disable_render,SIGNAL(clicked()),this,SLOT(onChangeRenderMode()));
    connect(ui->b_start_navigation,SIGNAL(clicked()),this,SLOT(onClickStartNavigation()));

    connect(filterSelectionForm,SIGNAL(activeFiltersChanged(QVector<int>,QVector<int>)),this,SLOT(activeFiltersChanged(QVector<int>,QVector<int>)));
    connect(ui->b_changeActiveFilters,SIGNAL(clicked()),this,SLOT(onClickChangeActiveFilters()));

    connect(this,SIGNAL(startPushBotController()),&pushBotController,SLOT(startProcessing()));
    connect(this,SIGNAL(stopPushBotController()),&pushBotController,SLOT(stopProcessing()));
}
void MainWindow::onUpdate()
{
    qint64 elapsed = timer.nsecsElapsed();
    timer.restart();

    bool debugMode = ui->rb_debug->isChecked();
    int orientIdx = ui->cb_show_orient->currentIndex();
    int speedIdx = ui->cb_show_speed->currentIndex();

    if(worker.isInitialized() && !ui->rb_disable_render->isChecked()) {

        if(debugMode) {
            quint32 time = worker.getMotionEnergy(speedIdx,orientIdx,motionEnergy);
            if(time != UINT32_MAX) {

                QImage &img1 = motionEnergy.toImage(0,1.0f);
                ui->l_img_energy->setImage(img1);
            }
        }

        worker.getOpticFlow(speed,dir,energy);

        //QFile file2("phase2.png");
        //file2.open(QIODevice::WriteOnly);
        //dir.toImage(-360,0).save(&file2,"PNG");
        //file2.close();
#ifdef DEBUG_INSERT_PROFILER_MARKS
        nvtxRangeId_t id = nvtxRangeStart("Flow 2 RGB");
#endif
        cudaFlowToRGB(speed.getGPUPtr(),dir.getGPUPtr(),gpuRgbImage,
                      DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_HEIGHT,
                      allSettings.at(activeSettingIndices.back()).speed_px_per_sec,
                      cudaStream);
        gpuErrchk(cudaMemcpyAsync(rgbImg.bits(),gpuRgbImage,
                                  DVS_RESOLUTION_WIDTH*DVS_RESOLUTION_HEIGHT*3,
                                  cudaMemcpyDeviceToHost,cudaStream));
        cudaStreamSynchronize(cudaStream);
#ifdef DEBUG_INSERT_PROFILER_MARKS
        nvtxRangeEnd(id);
#endif

        ui->l_img_flow->setImage(rgbImg);

        if(debugMode) {
            QImage &engImage = energy.toImage(0,1);
            ui->l_img_ctrl_2->setImage(engImage);
        }

        rgbImg.fill(Qt::white);
        QPainter painter(&rgbImg);

        QList<DVSEvent> ev = worker.getEventsInWindow(speedIdx);
        if(ev.length() > 0) {
            QPoint points[ev.length()];
            painter.setPen(QPen(Qt::black));
            for(int i = 0; i < ev.length(); i++) {
                points[i].setX(ev.at(i).x);
                points[i].setY(ev.at(i).y);
            }
            painter.drawPoints(points,ev.length());
        }

        if(debugMode) {
            painter.setPen(QPen(Qt::blue,1));
            painter.drawLine(DVS_RESOLUTION_WIDTH/2,0,DVS_RESOLUTION_WIDTH/2,DVS_RESOLUTION_HEIGHT);
            painter.setPen(QPen(Qt::blue,2));

            float fxL,fxR,fyL,fyR;
            QPoint p1L,p1R,p1C,p2L,p2R,p2C;

            //paint2.drawLine(DVS_RESOLUTION_WIDTH/2,0,DVS_RESOLUTION_WIDTH/2,DVS_RESOLUTION_HEIGHT);
            float maxLDraw = DVS_RESOLUTION_WIDTH/4.0f;
            float maxLength = allSettings.at(activeSettingIndices.back()).speed_px_per_sec; // Fastest speed as maximum
            p1L.setX(DVS_RESOLUTION_WIDTH/4);
            p1L.setY(DVS_RESOLUTION_HEIGHT/2);
            p1R.setX(DVS_RESOLUTION_WIDTH*3.0f/4.0f);
            p1R.setY(DVS_RESOLUTION_HEIGHT/2);
            p1C.setX(DVS_RESOLUTION_WIDTH/2);
            p1C.setY(DVS_RESOLUTION_HEIGHT*2/3);

            bool valid = pushBotController.getAvgSpeed(fxL,fyL,fxR,fyR);
            if(valid) {
                float lL = qSqrt(fxL*fxL+fyL*fyL);
                float scaleL = lL/maxLength;
                float lR = qSqrt(fxR*fxR+fyR*fyR);
                float scaleR = lR/maxLength;
                float lC = qAbs(lR-lL);
                float scaleC = lC/maxLength;

                if(scaleL > 0.01f) {
                    p2L = p1L + QPoint(maxLDraw*fxL/lL*scaleL,maxLDraw*fyL/lL*scaleL);
                    painter.drawLine(p1L,p2L);
                }
                if(scaleR > 0.01f) {
                    p2R = p1R + QPoint(maxLDraw*fxR/lR*scaleR,maxLDraw*fyR/lR*scaleR);
                    painter.drawLine(p1R,p2R);
                }
                if(scaleC > 0.01) {
                    p2C = p1C + QPoint(maxLDraw*(fxL-fxR)/lC*scaleC,maxLDraw*(fyL-fyR)/lC*scaleC);
                    painter.drawLine(p1C,p2C);
                }
                float ctrl = pushBotController.getCtrlOutput();
                p1C.setY(DVS_RESOLUTION_HEIGHT*1/3);
                p2C = p1C + QPoint(ctrl*2*maxLDraw/(PUSHBOT_VELOCITY_MAX-PUSHBOT_VELOCITY_MIN),0);
                painter.setPen(QPen(Qt::red,2));
                painter.drawLine(p1C,p2C);

            } else {
                painter.setPen(QPen(Qt::red,2));
                painter.drawRect(1,1,DVS_RESOLUTION_WIDTH-2,DVS_RESOLUTION_HEIGHT-2);
            }
        }

        painter.end();
        ui->l_img_events->setImage(rgbImg);

    }
    if(lastStatisticsUpdate.elapsed() > 1000.0f*1.0f/GUI_STAT_UPDATE_FPS) {
        quint32 evRec = 0, evDisc = 0;
        worker.getStats(evRec,evDisc,speedIdx);
        int globalSpeedIdx = activeSettingIndices.at(speedIdx);
        float p = 0;
        if(evRec > 0)
            p = 1- (float)evDisc/evRec;

        ui->l_proc_ratio->setText(QString("%1 %").arg(p*100,0,'g',4));
        ui->l_skip_ev_cnt->setNum((int)evDisc);
        ui->l_rec_ev_cnt->setNum((int)evRec);
        ui->l_ctrl_output->setText(QString("%1").arg(pushBotController.getCtrlOutput()));
        ui->l_time_per_slot->setText(QString("%1 us").arg(allSettings.at(globalSpeedIdx).timewindow_us/allSettings.at(globalSpeedIdx).temporalSteps));
        ui->l_timewindow->setText(QString("%1 us").arg(allSettings.at(globalSpeedIdx).timewindow_us));
        lastStatisticsUpdate.restart();
    }
    //qDebug("%llu %llu",elapsed,timer.nsecsElapsed());
    timer.restart();
}

void MainWindow::onPlaybackFinished()
{
    PRINT_DEBUG("PlaybackFinished");
    emit stopPushBotController();
    ui->b_start_playback->setText("Start");
    ui->tab_online->setEnabled(true);
    ui->gb_playback_settings->setEnabled(true);
    ui->b_changeActiveFilters->setEnabled(true);
    QMessageBox::information(this,"Information","Playback finished!");
}

void MainWindow::onClickStartPlayback()
{
    if(eDVSHandler.isWorking()) {
        PRINT_DEBUG("Stop Playback");
        emit stopPushBotController();
        stopAll();
        ui->b_start_playback->setText("Start");
        ui->tab_online->setEnabled(true);
        ui->gb_playback_settings->setEnabled(true);
        ui->b_changeActiveFilters->setEnabled(true);
    } else {
        PRINT_DEBUG("Start Playback");
        ui->l_img_flow->clear();
        ui->l_img_ctrl_2->clear();
        ui->l_img_events->clear();
        ui->b_start_playback->setText("Stop");
        ui->tab_online->setEnabled(false);
        ui->gb_playback_settings->setEnabled(false);
        ui->b_changeActiveFilters->setEnabled(false);
        QString file = ui->le_file_name_playback->text();
        float speed = ui->sb_play_speed->value()/100.0f;

        emit startPushBotController();
        emit playbackFile(file,speed);
    }
}

void MainWindow::onConnectionClosed(bool error)
{
    ui->b_connect->setText("Connect");
    ui->tab_playback->setEnabled(true);
    ui->gb_cmdline->setEnabled(false);
    ui->b_start_streaming->setEnabled(false);
    ui->b_reset->setEnabled(false);
    ui->gb_connect_settings->setEnabled(true);
    ui->b_start_navigation->setEnabled(false);
    ui->b_start_navigation->setText("Start navigation");
    ui->b_start_streaming->setText("Start streaming");

    if(error) {
        QMessageBox::critical(this,"Error","Connection closed unexpectedly!");
    }
}

void MainWindow::onConnectionResult(bool error)
{
    if(!error) {
        ui->b_connect->setText("Disconnect");
        ui->tab_playback->setEnabled(false);
        ui->gb_cmdline->setEnabled(true);
        ui->b_start_streaming->setEnabled(true);
        ui->b_start_navigation->setEnabled(true);
        ui->b_reset->setEnabled(true);
        ui->gb_connect_settings->setEnabled(false);
        ui->b_changeActiveFilters->setEnabled(false);
        ui->l_img_flow->clear();
        ui->l_img_ctrl_2->clear();
        ui->l_img_events->clear();
    } else {
        QMessageBox::critical(this,"Error","Failed to connect!");
    }
}

void MainWindow::onChangePlaybackFile()
{
    QString fileName = QFileDialog::getOpenFileName(this,
                       tr("Open Playback file"), "~", tr("Playback files (*.aedat)"));

    if(!fileName.isEmpty())
        ui->le_file_name_playback->setText(fileName);
}

void MainWindow::onLineRecived(QString answ)
{
    ui->te_comands->moveCursor (QTextCursor::End);
    ui->te_comands->insertPlainText (QString("Response: %1").arg(answ.append("\n")));
    ui->te_comands->moveCursor (QTextCursor::End);
}
void MainWindow::onCmdSent(QString cmd)
{
    ui->te_comands->moveCursor (QTextCursor::End);
    ui->te_comands->insertPlainText (QString("Command: %1").arg(cmd));
    ui->te_comands->moveCursor (QTextCursor::End);
}

void MainWindow::onClickStartStreaming()
{
    if(eDVSHandler.isStreaming())
        emit stopEventStreaming();
    else
        emit startEventStreaming();
}
void MainWindow::onStreamingStarted()
{
    ui->b_start_streaming->setText("Stop streaming");
    ui->b_reset->setEnabled(false);
    ui->b_start_navigation->setEnabled(true);
}

void MainWindow::onStreamingStopped()
{
    ui->b_start_streaming->setText("Start streaming");
    ui->b_reset->setEnabled(true);
    ui->b_start_navigation->setEnabled(false);
}

void MainWindow::onClickConnect()
{
    if(eDVSHandler.isWorking()) {
        stopAll();
        ui->b_connect->setText("Connect");
        ui->b_start_streaming->setText("Start streaming");
        ui->tab_playback->setEnabled(true);
        ui->gb_cmdline->setEnabled(false);
        ui->b_start_streaming->setEnabled(false);
        ui->b_start_navigation->setEnabled(false);
        ui->b_reset->setEnabled(false);
        ui->gb_connect_settings->setEnabled(true);
        ui->b_changeActiveFilters->setEnabled(true);
    } else {
        ui->te_comands->clear();
        emit connectToBot(ui->le_host->text());
    }
}
void MainWindow::onCmdEntered()
{
    if(eDVSHandler.isConnected()) {
        QString txt = ui->le_cmd_input->text();
        qDebug("%s",txt.toLocal8Bit().data());
        if(txt.trimmed().isEmpty())
            return;
        txt.append("\n");
        emit sendRawCmd(txt);
        ui->le_cmd_input->blockSignals(true);
        ui->le_cmd_input->clear();
        ui->le_cmd_input->blockSignals(false);
    }
}
void MainWindow::onClickReset()
{
    if(eDVSHandler.isConnected()) {
        emit reset();
    }
}

void MainWindow::onChangeThreshold(double v)
{
    worker.setEnergyThreshold(v);
}

void MainWindow::onChangePushbotP(double v)
{
    pushBotController.setP(v);
}

void MainWindow::onChangePushbotI(double v)
{
    pushBotController.setI(v);
}

void MainWindow::onChangePushbotD(double v)
{
    pushBotController.setD(v);
}

void MainWindow::onChangeRenderMode()
{
    if(ui->rb_debug->isChecked()) {
        ui->l_img_events->show();
        ui->l_img_flow->show();
        ui->l_img_energy->show();
        ui->l_img_ctrl_2->show();
    } else if(ui->rb_disable_render->isChecked()) {
        ui->l_img_events->hide();
        ui->l_img_flow->hide();
        ui->l_img_energy->hide();
        ui->l_img_ctrl_2->hide();
    } else {
        ui->l_img_events->show();
        ui->l_img_flow->show();
        ui->l_img_energy->hide();
        ui->l_img_ctrl_2->hide();
    }
}

void MainWindow::onClickChangeActiveFilters()
{
    filterSelectionForm->show(allOrientations,
                              allSettings,
                              activeOrientationIndices,
                              activeSettingIndices);
}

void MainWindow::activeFiltersChanged(QVector<int> activeOrientationIndices,
                                      QVector<int> activeSettingIndices)
{
    this->activeOrientationIndices = activeOrientationIndices;
    this->activeSettingIndices = activeSettingIndices;
    setupActiveFilters();
}

void MainWindow::setupActiveFilters()
{

    QVector<float> activeOrientations;
    Q_FOREACH(int i,activeOrientationIndices) {
        activeOrientations.append(allOrientations.at(i));
    }
    QVector<FilterSettings> activeSettings;
    Q_FOREACH(int i,activeSettingIndices) {
        activeSettings.append(allSettings.at(i));
    }

    ui->cb_show_speed->clear();
    Q_FOREACH(FilterSettings fs, activeSettings) {
        ui->cb_show_speed->addItem(QString("%1").arg(fs.speed_px_per_sec));
    }
    ui->cb_show_orient->clear();
    Q_FOREACH(float o, activeOrientations) {
        ui->cb_show_orient->addItem(QString("%1").arg(RAD2DEG(o)));
    }
    pushBotController.setup(activeSettings,activeOrientations);
    worker.setComputationParameters(activeSettings,activeOrientations);
}

void MainWindow::onClickStartNavigation()
{
    //if(!eDVSHandler.isStreaming()) {
    //    qWarning("Not streaming! Navigation not possible!");
    //    return;
    //}
    if(!pushBotController.isProcessing()) {
        ui->b_start_navigation->setText("Stop navigation");
        ui->b_start_streaming->setEnabled(false);
        emit startPushBotController();
    } else {
        ui->b_start_navigation->setText("Start navigation");
        ui->b_start_streaming->setEnabled(true);
        emit stopPushBotController();
    }
}

void MainWindow::closeEvent(QCloseEvent *bar)
{
    stopAll();
    QMainWindow::closeEvent(bar);
}

void MainWindow::stopAll()
{
    if(pushBotController.isProcessing())
        emit stopPushBotController();

    // Wait for pushbot to stop
    while (pushBotController.isProcessing()) {
        QThread::usleep(1000);
    }

    if(eDVSHandler.isWorking())
        emit eDVSInterfaceStopWork();

    // Wait for eDVSInterface to stop
    while (eDVSHandler.isWorking()) {
        QThread::usleep(1000);
    }

}
