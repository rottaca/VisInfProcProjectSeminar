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
    initUI();
    initSystem();
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
    Q_FOREACH(QSerialPortInfo port, QSerialPortInfo::availablePorts()) {
            ui->cb_ports->addItem(port.portName());
        }
}

void MainWindow::initSystem()
{
    gpuErrchk(cudaSetDevice(0));
    cudaStreamCreate(&cudaStream);

    settings.append(FilterSettings::getSettings(FilterSettings::SPEED_25));
    //settings.append(FilterSettings::getSettings(FilterSettings::SPEED_12_5));
    //settings.append(FilterSettings::getSettings(FilterSettings::SPEED_50));

    orientations.append(qDegreesToRadians(0.0f));
    orientations.append(qDegreesToRadians(90.0f));

    worker = new Worker();
    dvsEventHandler.setWorker(worker);
}

void MainWindow::initSignalsAndSlots()
{
    connect(&updateTimer,SIGNAL(timeout()),this,SLOT(OnUpdate()));
    connect(this,SIGNAL(startProcessing()),worker,SLOT(start()));
    connect(&dvsEventHandler,SIGNAL(OnPlaybackFinished()),this,SLOT(OnPlaybackFinished()));
    connect(ui->b_browse_play_file,SIGNAL(clicked()),this,SLOT(OnChangePlaybackFile()));
    connect(ui->b_start_playback,SIGNAL(clicked()),this,SLOT(OnClickStartPlayback()));
}

void MainWindow::OnUpdate()
{
    if(worker->getIsProcessing()){
        long time = worker->getMotionEnergy(0,0,oppMoEnergy1);
        if(time != -1){
            worker->getMotionEnergy(0,1,oppMoEnergy2);
            worker->getOpticFlow(flowX,flowY);
            flowX.setCudaStream(cudaStream);
            flowY.setCudaStream(cudaStream);
            oppMoEnergy1.setCudaStream(cudaStream);
            oppMoEnergy2.setCudaStream(cudaStream);

            QImage img1 = oppMoEnergy1.toImage(-0.5,0.5);
            ui->label_1->setPixmap(QPixmap::fromImage(img1));
            QImage img2 = oppMoEnergy2.toImage(-0.5,0.5);
            ui->label_2->setPixmap(QPixmap::fromImage(img2));

            // TODO Move to GPU
            float* ptrOfV1 = flowX.getCPUPtr();
            float* ptrOfV2 = flowY.getCPUPtr();

            int sz = 128;
            int imgScale = 4;
            float maxL = 0.3;
            int spacing = 2;
            int length = 20;
            float minPercentage = 0.3;

            QVector<QLine> lines;
            QVector<QPoint> points;
            for(int y = 0; y < DVS_RESOLUTION_HEIGHT; y+=spacing){
                for(int x = 0; x < DVS_RESOLUTION_WIDTH; x+=spacing){
                    int i = x + y*DVS_RESOLUTION_WIDTH;
                    QLine line;
                    float l = qSqrt(ptrOfV1[i]*ptrOfV1[i] + ptrOfV2[i]*ptrOfV2[i]);
                    float percentage = qMin(1.0f,l/maxL);

                    if(percentage > minPercentage){
                        int x2 = x + percentage*length*ptrOfV1[i];
                        int y2 = y + percentage*length*ptrOfV2[i];
                        //qDebug(QString("%1 %2 %3 %4 %5 %6").arg(x).arg(y).arg(x2).arg(y2).arg(percentage).arg(l).toLocal8Bit());
                        line.setLine(x*imgScale,y*imgScale,x2*imgScale,y2*imgScale);
                        lines.append(line);
                        points.append(QPoint(x*imgScale,y*imgScale));
                    }
                }
            }
            //qDebug(QString("%1").arg(lines.length()).toLocal8Bit());

            QImage imgFlow(imgScale*sz,imgScale*sz,QImage::Format_RGB888);
            imgFlow.fill(Qt::white);
            QPainter painter1(&imgFlow);
            QPen pointpen(Qt::red);
            pointpen.setWidth(2);
            painter1.setPen(pointpen);
            painter1.drawPoints(points);
            QPen linepen(Qt::black);
            painter1.setPen(linepen);
            painter1.drawLines(lines);
            painter1.end();
            ui->label_3->setPixmap(QPixmap::fromImage(imgFlow));
        }else{
            //qDebug("No new data available!");
        }

        QVector<DVSEventHandler::DVSEvent> ev = worker->getEventsInWindow(0);
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
        ui->label_eventwindow->setPixmap(QPixmap::fromImage(img));

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

void MainWindow::OnPlaybackFinished()
{
    worker->stopProcessing();
    ui->b_start_playback->setText("Start");
}

void MainWindow::OnClickStartPlayback(){
    if(worker->getIsProcessing()){
        worker->stopProcessing();
        dvsEventHandler.abort();
        ui->b_start_playback->setText("Start");

    }else{

        ui->b_start_playback->setText("Stop");
        QString file = ui->le_file_name_playback->text();
        float speed = ui->sb_play_speed->value()/100.0f;

        worker->createOpticFlowEstimator(settings,orientations);
        worker->start();
        dvsEventHandler.playbackFile(file,speed);
    }
}

void MainWindow::OnChangePlaybackFile()
{
    QString fileName = QFileDialog::getOpenFileName(this,
        tr("Open Playback file"), "~", tr("Playback files (*.aedat)"));

    if(!fileName.isEmpty())
        ui->le_file_name_playback->setText(fileName);
}
