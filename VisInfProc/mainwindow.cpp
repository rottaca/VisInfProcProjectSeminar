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

#include "settings.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    initSystem();
    initUI();
    initSignalsAndSlots();

    updateTimer.start(1000/FPS);
}
MainWindow::~MainWindow()
{
    delete ui;
    delete worker;
    cudaStreamDestroy(cudaStream);
}

void MainWindow::initUI()
{
    ui->setupUi(this);
    lastStatisticsUpdate.start();

    ui->cb_show_speed->clear();
    Q_FOREACH(FilterSettings fs, settings){
        ui->cb_show_speed->addItem(QString("%1").arg(fs.speed_px_per_sec));
    }
    ui->cb_show_orient->clear();
    Q_FOREACH(float o, orientations){
        ui->cb_show_orient->addItem(QString("%1").arg(qRadiansToDegrees(o)));
    }

}

void MainWindow::initSystem()
{
    gpuErrchk(cudaSetDevice(0));
    cudaStreamCreate(&cudaStream);

    //settings.append(FilterSettings::getSettings(FilterSettings::SPEED_12_5));
    settings.append(FilterSettings::getSettings(FilterSettings::SPEED_21));
    settings.append(FilterSettings::getSettings(FilterSettings::SPEED_46));

    orientations.append(qDegreesToRadians(0.0f));
    orientations.append(qDegreesToRadians(180.0f));
    orientations.append(qDegreesToRadians(-90.0f));
    orientations.append(qDegreesToRadians(90.0f));

    //orientations.append(qDegreesToRadians(45.0f));
    //orientations.append(qDegreesToRadians(-45.0f));
    //orientations.append(qDegreesToRadians(-135.0f));
    //orientations.append(qDegreesToRadians(135.0f));

    worker = new Worker();
    eDVSHandler.setWorker(worker);
}

void MainWindow::initSignalsAndSlots()
{
    connect(&updateTimer,SIGNAL(timeout()),this,SLOT(onUpdate()));

    connect(this,SIGNAL(startProcessing()),worker,SLOT(start()));
    connect(this,SIGNAL(sendRawCmd(QString)),&eDVSHandler,SLOT(sendRawCmd(QString)));

    connect(&eDVSHandler,SIGNAL(onCmdSent(QString)),this,SLOT(onCmdSent(QString)));
    connect(&eDVSHandler,SIGNAL(onLineRecived(QString)),this,SLOT(onLineRecived(QString)));
    connect(&eDVSHandler,SIGNAL(onPlaybackFinished()),this,SLOT(onPlaybackFinished()));
    connect(&eDVSHandler,SIGNAL(onConnectionResult(bool)),this,SLOT(onConnectionResult(bool)));

    connect(ui->b_browse_play_file,SIGNAL(clicked()),this,SLOT(onChangePlaybackFile()));
    connect(ui->b_start_playback,SIGNAL(clicked()),this,SLOT(onClickStartPlayback()));
    connect(ui->b_connect,SIGNAL(clicked()),this,SLOT(onClickConnect()));
    connect(ui->b_start_streaming,SIGNAL(clicked()),this,SLOT(onClickStartStreaming()));
    connect(ui->le_cmd_input,SIGNAL(editingFinished()),this,SLOT(onCmdEntered()));
}

void MainWindow::onUpdate()
{
    if(worker->getIsProcessing()){
        int orientIdx = ui->cb_show_orient->currentIndex();
        int speedIdx = ui->cb_show_speed->currentIndex();

        long time = worker->getMotionEnergy(speedIdx,orientIdx,oppMoEnergy1);
        if(time != -1){
            worker->getOpticFlow(flowX,flowY);
            flowX.setCudaStream(cudaStream);
            flowY.setCudaStream(cudaStream);
            oppMoEnergy1.setCudaStream(cudaStream);
            oppMoEnergy2.setCudaStream(cudaStream);

            QImage img1 = oppMoEnergy1.toImage(0,1.0f);
            ui->l_motion->setPixmap(QPixmap::fromImage(img1));

            QImage rgbFlow(DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_HEIGHT,QImage::Format_RGB888);
            char* gpuImage;
            gpuErrchk(cudaMalloc(&gpuImage,DVS_RESOLUTION_WIDTH*DVS_RESOLUTION_HEIGHT*3));


            cudaFlowToRGB(flowX.getGPUPtr(),flowY.getGPUPtr(),gpuImage,
                          DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_HEIGHT,1,cudaStream);
            gpuErrchk(cudaMemcpyAsync(rgbFlow.bits(),gpuImage,
                                      DVS_RESOLUTION_WIDTH*DVS_RESOLUTION_HEIGHT*3,
                                      cudaMemcpyDeviceToHost,cudaStream));
            cudaStreamSynchronize(cudaStream);
            ui->l_flow->setPixmap(QPixmap::fromImage(rgbFlow));

        }else{
            //qDebug("No new data available!");
        }

        QVector<SerialeDVSInterface::DVSEvent> ev = worker->getEventsInWindow(0);
        QPoint points[ev.length()];
        for(int i = 0; i < ev.length(); i++){
            points[i].setX(ev.at(i).posX);
            points[i].setY(ev.at(i).posY);
        }
        QImage img(DVS_RESOLUTION_HEIGHT,DVS_RESOLUTION_WIDTH,QImage::Format_RGB888);
        img.fill(Qt::white);
        QPainter painter(&img);
        painter.drawPoints(points,ev.length());
        painter.end();
        ui->l_events->setPixmap(QPixmap::fromImage(img));

        if(lastStatisticsUpdate.elapsed() > 500){
            long evRec, evDisc;
            worker->getStats(evRec,evDisc);
            float p = 0;
            if(evRec > 0)
                p = 1- (float)evDisc/evRec;

            ui->l_proc_ratio->setText(QString("%1 %").arg(p*100,0,'g',4));
            ui->l_skip_ev_cnt->setNum((int)evDisc);
            ui->l_rec_ev_cnt->setNum((int)evRec);
            ui->l_timestamp->setNum((int)time);
            lastStatisticsUpdate.restart();
        }
    }
}

void MainWindow::onPlaybackFinished()
{
    qDebug("PlaybackFinished");
    ui->b_start_playback->setText("Start");
    ui->tab_online->setEnabled(true);
}

void MainWindow::onClickStartPlayback(){
    if(eDVSHandler.isWorking()){
        qDebug("Stop Playback");
        eDVSHandler.stopWork();
        ui->b_start_playback->setText("Start");
        ui->tab_online->setEnabled(true);
    }else{
        qDebug("Start Playback");
        ui->b_start_playback->setText("Stop");
        ui->tab_online->setEnabled(false);
        QString file = ui->le_file_name_playback->text();
        float speed = ui->sb_play_speed->value()/100.0f;

        worker->createOpticFlowEstimator(settings,orientations);
        eDVSHandler.playbackFile(file,speed);
    }
}

void MainWindow::onConnectionResult(bool failed)
{
    if(!failed){
        ui->b_connect->setText("Disconnect");
        ui->tab_playback->setEnabled(false);
        ui->gb_cmdline->setEnabled(true);
        ui->b_start_streaming->setEnabled(true);
    }else{
        qDebug("Failed to connect!");
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
    ui->te_comands->insertPlainText (QString("Answer: %1").arg(answ));
    ui->te_comands->moveCursor (QTextCursor::End);
}
void MainWindow::onCmdSent(QString cmd)
{
    ui->te_comands->moveCursor (QTextCursor::End);
    ui->te_comands->insertPlainText (QString("Cmd: %1").arg(cmd));
    ui->te_comands->moveCursor (QTextCursor::End);
}

void MainWindow::onClickStartStreaming()
{
    if(eDVSHandler.isStreaming()){
        ui->b_start_streaming->setText("Start Streaming");
        eDVSHandler.stopEventStreaming();
        worker->stopProcessing();
    }else{
        ui->b_start_streaming->setText("Stop Streaming");
        eDVSHandler.startEventStreaming();
        worker->createOpticFlowEstimator(settings,orientations);
        worker->start();
    }
}

void MainWindow::onClickConnect()
{
    if(eDVSHandler.isWorking()){
        eDVSHandler.stopWork();
        ui->b_connect->setText("Connect");
        ui->tab_playback->setEnabled(true);
        ui->gb_cmdline->setEnabled(false);
        ui->b_start_streaming->setEnabled(false);
    }else{
        worker->createOpticFlowEstimator(settings,orientations);
        eDVSHandler.connectToBot(ui->le_host->text(),ui->sb_port->value());
    }
}
void MainWindow::onCmdEntered()
{
    if(eDVSHandler.isConnected()){
        QString txt = ui->le_cmd_input->text();
        txt.append("\n");
        eDVSHandler.sendRawCmd(txt);
    }
}
