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

    updateTimer.start(1000/FPS);
}
MainWindow::~MainWindow()
{
    delete ui;
    cudaStreamDestroy(cudaStream);
}

void MainWindow::initUI()
{
    ui->setupUi(this);
    lastStatisticsUpdate.start();

    ui->cb_show_speed->clear();
    Q_FOREACH(FilterSettings fs, settings)
        {
            ui->cb_show_speed->addItem(QString("%1").arg(fs.speed_px_per_sec));
        }
    ui->cb_show_orient->clear();
    Q_FOREACH(float o, orientations)
        {
            ui->cb_show_orient->addItem(QString("%1").arg(qRadiansToDegrees(o)));
        }

}

void MainWindow::initSystem()
{
    gpuErrchk(cudaSetDevice(0));
    cudaStreamCreate(&cudaStream);

    settings.append(FilterSettings::getSettings(FilterSettings::SPEED_1));
    settings.append(FilterSettings::getSettings(FilterSettings::SPEED_2));
    settings.append(FilterSettings::getSettings(FilterSettings::SPEED_3));
    //settings.append(FilterSettings::getSettings(FilterSettings::SPEED_4));
    //settings.append(FilterSettings::getSettings(FilterSettings::SPEED_5));

    // Sort list of settings by speed for interpolation
    std::sort(settings.begin(),settings.end(),sortSettingsBySpeed);

    orientations.append(qDegreesToRadians(0.0f));
    orientations.append(qDegreesToRadians(180.0f));
    orientations.append(qDegreesToRadians(-90.0f));
    orientations.append(qDegreesToRadians(90.0f));

//    orientations.append(qDegreesToRadians(45.0f));
//    orientations.append(qDegreesToRadians(-45.0f));
//    orientations.append(qDegreesToRadians(-135.0f));
//    orientations.append(qDegreesToRadians(135.0f));

    pushBotController.setup(settings,orientations);
    worker.setComputationParameters(settings,orientations);

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

    connect(&eDVSHandler,SIGNAL(onCmdSent(QString)),this,SLOT(onCmdSent(QString)));
    connect(&eDVSHandler,SIGNAL(onLineRecived(QString)),this,SLOT(onLineRecived(QString)));
    connect(&eDVSHandler,SIGNAL(onPlaybackFinished()),this,SLOT(onPlaybackFinished()));
    connect(&eDVSHandler,SIGNAL(onConnectionResult(bool)),this,SLOT(onConnectionResult(bool)));
    connect(&eDVSHandler,SIGNAL(onConnectionClosed(bool)),this,SLOT(onConnectionClosed(bool)));

    connect(ui->b_browse_play_file,SIGNAL(clicked()),this,SLOT(onChangePlaybackFile()));
    connect(ui->b_start_playback,SIGNAL(clicked()),this,SLOT(onClickStartPlayback()));
    connect(ui->b_connect,SIGNAL(clicked()),this,SLOT(onClickConnect()));
    connect(ui->b_start_streaming,SIGNAL(clicked()),this,SLOT(onClickStartStreaming()));
    connect(ui->le_cmd_input,SIGNAL(editingFinished()),this,SLOT(onCmdEntered()));
    connect(ui->b_reset,SIGNAL(clicked()),this,SLOT(onClickReset()));
}

void MainWindow::onUpdate()
{
    // TODO Speedup visualization code
    if(worker.isInitialized() && !ui->cb_disable_render->isChecked())
        {
            int orientIdx = ui->cb_show_orient->currentIndex();
            int speedIdx = ui->cb_show_speed->currentIndex();

            long time = worker.getMotionEnergy(speedIdx,orientIdx,oppMoEnergy1);
            if(time != -1)
                {
                    if(ui->cb_debug->isChecked())
                        {
//                Buffer3D en;
//                worker.getConvBuffer(speedIdx,orientIdx,0,en);
//                QImage imgConv = en.toImageXZ(63);
//                ui->l_img_debug->setPixmap(QPixmap::fromImage(imgConv));
                            worker.getOpticFlow(speed,dir,energy);
                            QImage rgbFlow(DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_HEIGHT,QImage::Format_RGB888);
                            char* gpuImage;
                            gpuErrchk(cudaMalloc(&gpuImage,DVS_RESOLUTION_WIDTH*DVS_RESOLUTION_HEIGHT*3));

                            cudaFlowToRGB(speed.getGPUPtr(),dir.getGPUPtr(),gpuImage,
                                          DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_HEIGHT,60,cudaStream);
                            gpuErrchk(cudaMemcpyAsync(rgbFlow.bits(),gpuImage,
                                                      DVS_RESOLUTION_WIDTH*DVS_RESOLUTION_HEIGHT*3,
                                                      cudaMemcpyDeviceToHost,cudaStream));
                            cudaStreamSynchronize(cudaStream);
                            gpuErrchk(cudaFree(gpuImage));
                            ui->l_img_debug->setPixmap(QPixmap::fromImage(rgbFlow));

                            QImage engImage = energy.toImage(0,1);
                            ui->l_ctrl_2->setPixmap(QPixmap::fromImage(engImage));
                        }

                    worker.getOpticFlowEnergy(energy,dir,speedIdx);
                    energy.setCudaStream(cudaStream);
                    dir.setCudaStream(cudaStream);
                    oppMoEnergy1.setCudaStream(cudaStream);
                    oppMoEnergy2.setCudaStream(cudaStream);

                    QImage img1 = oppMoEnergy1.toImage(0,1.0f);
                    ui->l_motion->setPixmap(QPixmap::fromImage(img1));

                    QImage rgbFlow(DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_HEIGHT,QImage::Format_RGB888);
                    char* gpuImage;
                    gpuErrchk(cudaMalloc(&gpuImage,DVS_RESOLUTION_WIDTH*DVS_RESOLUTION_HEIGHT*3));

                    cudaFlowToRGB(energy.getGPUPtr(),dir.getGPUPtr(),gpuImage,
                                  DVS_RESOLUTION_WIDTH,DVS_RESOLUTION_HEIGHT,1,cudaStream);
                    gpuErrchk(cudaMemcpyAsync(rgbFlow.bits(),gpuImage,
                                              DVS_RESOLUTION_WIDTH*DVS_RESOLUTION_HEIGHT*3,
                                              cudaMemcpyDeviceToHost,cudaStream));
                    cudaStreamSynchronize(cudaStream);
                    gpuErrchk(cudaFree(gpuImage));
                    ui->l_flow->setPixmap(QPixmap::fromImage(rgbFlow));

                }
            else
                {
                    //qDebug("No new data available!");
                }

            if(ui->cb_debug->isChecked())
                {
                    float fxL,fxR,fyL,fyR;
                    QImage imgAvg(DVS_RESOLUTION_HEIGHT,DVS_RESOLUTION_WIDTH,QImage::Format_RGB888);
                    imgAvg.fill(Qt::white);
                    QPainter paint2(&imgAvg);
                    QPoint p1L,p1R,p1C,p2L,p2R,p2C;

                    //paint2.drawLine(DVS_RESOLUTION_WIDTH/2,0,DVS_RESOLUTION_WIDTH/2,DVS_RESOLUTION_HEIGHT);
                    float maxLDraw = DVS_RESOLUTION_WIDTH/4;
                    float maxLength = 60;
                    p1L.setX(DVS_RESOLUTION_WIDTH/4);
                    p1L.setY(DVS_RESOLUTION_HEIGHT/2);
                    p1R.setX(DVS_RESOLUTION_WIDTH*3.0f/4);
                    p1R.setY(DVS_RESOLUTION_HEIGHT/2);
                    p1C.setX(DVS_RESOLUTION_WIDTH/2);
                    p1C.setY(DVS_RESOLUTION_HEIGHT*2/3);

                    pushBotController.getAvgSpeed(fxL,fyL,fxR,fyR);
                    float lL = qSqrt(fxL*fxL+fyL*fyL);
                    float scaleL = lL/maxLength;
                    float lR = qSqrt(fxR*fxR+fyR*fyR);
                    float scaleR = lR/maxLength;
                    float lC = qAbs(lR-lL);
                    float scaleC = lC/maxLength;
                    QPen black(Qt::black);
                    paint2.setPen(black);
                    QRect rectL(0,0,DVS_RESOLUTION_WIDTH/2,DVS_RESOLUTION_HEIGHT);
                    QRect rectR(DVS_RESOLUTION_WIDTH/2,0,DVS_RESOLUTION_WIDTH/2,DVS_RESOLUTION_HEIGHT);

                    QColor c;
                    if(scaleL > 0.01)
                        {
                            p2L = p1L + QPoint(maxLDraw*fxL/lL*scaleL,maxLDraw*fyL/lL*scaleL);
                            int angle = ((int)(atan2(fyL,fxL)*180/M_PI + 360) % 360);

                            c.setHsv(angle,(int)qMin(255.f,255*scaleL),255);
                            paint2.fillRect(rectL,QBrush(c));
                            paint2.drawLine(p1L,p2L);
                        }
                    if(scaleR > 0.01)
                        {
                            p2R = p1R + QPoint(maxLDraw*fxR/lR*scaleR,maxLDraw*fyR/lR*scaleR);
                            int angle = ((int)(atan2(fyR,fxR)*180/M_PI + 360) % 360);

                            c.setHsv(angle,(int)qMin(255.f,255*scaleR),255);
                            paint2.fillRect(rectR,QBrush(c));
                            paint2.drawLine(p1R,p2R);
                        }
                    if(scaleC > 0.01)
                        {
                            p2C = p1C + QPoint(maxLDraw*(fxL-fxR)/lC*scaleC,maxLDraw*(fyL-fyR)/lC*scaleC);
                            paint2.drawLine(p1C,p2C);
                        }

                    paint2.end();
                    ui->l_ctrl_1->setPixmap(QPixmap::fromImage(imgAvg));
                }
            QList<eDVSInterface::DVSEvent> ev = worker.getEventsInWindow(speedIdx);
            if(ev.length() > 0)
                {
                    QPoint points[ev.length()];
                    for(int i = 0; i < ev.length(); i++)
                        {
                            points[i].setX(ev.at(i).posX);
                            points[i].setY(ev.at(i).posY);
                        }
                    QImage img(DVS_RESOLUTION_HEIGHT,DVS_RESOLUTION_WIDTH,QImage::Format_RGB888);
                    img.fill(Qt::white);
                    QPainter painter(&img);
                    painter.drawPoints(points,ev.length());
                    painter.end();
                    ui->l_events->setPixmap(QPixmap::fromImage(img));
                }
            if(lastStatisticsUpdate.elapsed() > 500)
                {
                    long evRec, evDisc;
                    worker.getStats(evRec,evDisc);
                    float p = 0;
                    if(evRec > 0)
                        p = 1- (float)evDisc/evRec;

                    ui->l_proc_ratio->setText(QString("%1 %").arg(p*100,0,'g',4));
                    ui->l_skip_ev_cnt->setNum((int)evDisc);
                    ui->l_rec_ev_cnt->setNum((int)evRec);
                    ui->l_timestamp->setNum((int)time);
                    ui->l_ctrl_output->setText(QString("%1").arg(pushBotController.getCtrlOutput()));
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

void MainWindow::onClickStartPlayback()
{
    if(eDVSHandler.isWorking())
        {
            qDebug("Stop Playback");
            eDVSHandler.stopWork();
            ui->b_start_playback->setText("Start");
            ui->tab_online->setEnabled(true);
        }
    else
        {
            qDebug("Start Playback");
            ui->b_start_playback->setText("Stop");
            ui->tab_online->setEnabled(false);
            QString file = ui->le_file_name_playback->text();
            float speed = ui->sb_play_speed->value()/100.0f;

            eDVSHandler.playbackFile(file,speed);
        }
}

void MainWindow::onConnectionClosed(bool error)
{
    ui->b_connect->setText("Connect");
    ui->tab_playback->setEnabled(true);
    ui->gb_cmdline->setEnabled(false);
    ui->b_start_streaming->setEnabled(false);
    ui->b_start_streaming->setText("Start");

    if(error)
        {
            QMessageBox::critical(this,"Error","Connection closed unexpectedly!");
        }
}

void MainWindow::onConnectionResult(bool error)
{
    if(!error)
        {
            ui->b_connect->setText("Disconnect");
            ui->tab_playback->setEnabled(false);
            ui->gb_cmdline->setEnabled(true);
            ui->b_start_streaming->setEnabled(true);
            ui->b_reset->setEnabled(true);
        }
    else
        {
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
        {
            ui->b_start_streaming->setText("Start");
            ui->b_reset->setEnabled(true);
            emit stopEventStreaming();
        }
    else
        {
            ui->b_start_streaming->setText("Stop");
            ui->b_reset->setEnabled(false);
            emit startEventStreaming();
        }
}

void MainWindow::onClickConnect()
{
    if(eDVSHandler.isWorking())
        {
            eDVSHandler.stopWork();
            ui->b_connect->setText("Connect");
            ui->tab_playback->setEnabled(true);
            ui->gb_cmdline->setEnabled(false);
            ui->b_start_streaming->setEnabled(false);
            ui->b_reset->setEnabled(false);
        }
    else
        {
            ui->te_comands->clear();
            eDVSHandler.connectToBot(ui->le_host->text(),ui->sb_port->value());
        }
}
void MainWindow::onCmdEntered()
{
    if(eDVSHandler.isConnected())
        {
            QString txt = ui->le_cmd_input->text();
            if(txt.trimmed().isEmpty())
                return;
            txt.append("\n");
            emit sendRawCmd(txt);
            ui->le_cmd_input->clear();
        }
}
void MainWindow::onClickReset()
{
    if(eDVSHandler.isConnected())
        {
            emit reset();
        }
}
