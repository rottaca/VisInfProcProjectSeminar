#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "cuda_helper.h"
#include <cuda_runtime.h>

#include <QtMath>
#include <QList>
#include <QPainter>
#include <QPoint>
#include <QLine>
#include <QtMath>

#include "buffer1d.h"
#include "buffer2d.h"
#include "buffer3d.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    gpuErrchk(cudaSetDevice(0));

    worker = new Worker();
    dvsEventHandler.setWorker(worker);
    FilterSettings fset = FilterSettings::getSettings(FilterSettings::SPEED_25);

    settings.append(fset);
    orientations.append(qDegreesToRadians(0.0f));
    orientations.append(qDegreesToRadians(90.0f));
    worker->createOpticFlowEstimator(settings,orientations);

    connect(&updateTimer,SIGNAL(timeout()),this,SLOT(OnUpdate()));
    connect(this,SIGNAL(startProcessing()),worker,SLOT(start()));
    connect(&dvsEventHandler,SIGNAL(OnPlaybackFinished()),this,SLOT(OnPlaybackFinished()));

    worker->start();

    dvsEventHandler.PlayBackFile("/tausch/BottiBot/dvs128_towers_take_1_2016-12-22.aedat");
    //dvsEventHandler.PlayBackFile("/tausch/BottiBot/dvs128_wall_take_2_2016-12-22.aedat");
    //dvsEventHandler.PlayBackFile("/tausch/scale4/mnist_0_scale04_0550.aedat");

    updateTimer.start(1000/FPS);
}
MainWindow::~MainWindow()
{
    delete ui;
}
void MainWindow::OnUpdate()
{
    if(worker->getIsProcessing()){
        long time = worker->getMotionEnergy(0,0,oppMoEnergy1);
        if(time != -1){
            worker->getMotionEnergy(0,1,oppMoEnergy2);
            //worker->getOpticFlow(flowX,flowY);

            QImage img1 = oppMoEnergy1.toImage(-0.3,0.3);
            ui->label_1->setPixmap(QPixmap::fromImage(img1));
            QImage img2 = oppMoEnergy2.toImage(-0.3,0.3);
            ui->label_2->setPixmap(QPixmap::fromImage(img2));

            ui->l_timestamp->setText(QString("%1").arg(time));

//            // TODO Move to GPU
//            double* ptrOfV1 = flowX.getCPUPtr();
//            double* ptrOfV2 = flowY.getCPUPtr();

//            int sz = 128;
//            int imgScale = 4;
//            double maxL = 0.3;
//            int spacing = 2;
//            int length = 10;
//            double minPercentage = 0.1;

//            QVector<QLine> lines;
//            QVector<QPoint> points;
//            for(int y = 0; y < 128; y+=spacing){
//                for(int x = 0; x < 128; x+=spacing){
//                    int i = x + y*128;
//                    QLine line;
//                    double l = qSqrt(ptrOfV1[i]*ptrOfV1[i] + ptrOfV2[i]*ptrOfV2[i]);
//                    double percentage = qMin(1.0,l/maxL);

//                    if(percentage > minPercentage){
//                        int x2 = x + percentage*length*ptrOfV1[i];
//                        int y2 = y + percentage*length*ptrOfV2[i];
//                        //qDebug(QString("%1 %2 %3 %4 %5 %6").arg(x).arg(y).arg(x2).arg(y2).arg(percentage).arg(l).toLocal8Bit());
//                        line.setLine(x*imgScale,y*imgScale,x2*imgScale,y2*imgScale);
//                        lines.append(line);
//                        points.append(QPoint(x*imgScale,y*imgScale));
//                    }
//                }
//            }
//            //qDebug(QString("%1").arg(lines.length()).toLocal8Bit());

//            QImage imgFlow(imgScale*sz,imgScale*sz,QImage::Format_RGB888);
//            imgFlow.fill(Qt::white);
//            QPainter painter1(&imgFlow);
//            QPen pointpen(Qt::red);
//            pointpen.setWidth(2);
//            painter1.setPen(pointpen);
//            painter1.drawPoints(points);
//            QPen linepen(Qt::black);
//            painter1.setPen(linepen);
//            painter1.drawLines(lines);
//            painter1.end();
//            ui->label_3->setPixmap(QPixmap::fromImage(imgFlow));
        }else{
            //qDebug("No new data available!");
        }

        QVector<DVSEventHandler::DVSEvent> ev = worker->getEventsInWindow(0);
        QPoint points[ev.length()];
        for(int i = 0; i < ev.length(); i++){
            points[i].setX(ev.at(i).posX);
            points[i].setY(ev.at(i).posY);
        }
        QImage img(128,128,QImage::Format_RGB888);
        img.fill(Qt::white);
        QPainter painter(&img);
        painter.drawPoints(points,ev.length());
        painter.end();
        ui->label_eventwindow->setPixmap(QPixmap::fromImage(img));

        int evRec, evDisc;
        worker->getStats(evRec,evDisc);
        float p = 0;
        if(evRec > 0)
            p = 1- (float)evDisc/evRec;

        ui->l_proc_ratio->setText(QString("%1 %").arg(p*100,0,'g',4));
        ui->l_proc_ev_cnt->setNum(evRec-evDisc);
        ui->l_rec_ev_cnt->setNum(evRec);
    }
}
void MainWindow::OnNewEvent(DVSEventHandler::DVSEvent e)
{
    worker->nextEvent(e);
}
void MainWindow::OnPlaybackFinished()
{
    worker->stopProcessing();
    //cudaProfilerStop();
    //worker->createOpticFlowEstimator(settings,orientations);
    //worker->start();

    //dvsEventHandler.PlayBackFile("/tausch/BottiBot/dvs128_towers_take_1_2016-12-22.aedat");
}
