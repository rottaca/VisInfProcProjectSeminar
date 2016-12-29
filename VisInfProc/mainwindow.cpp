#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "cuda_helper.h"

#include <QtMath>
#include <QList>
#include <QPainter>
#include <QPoint>

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

    QList<float> orients;
    QList<FilterSettings> fsettings;
    fsettings.append(fset);
    orients.append(qDegreesToRadians(0.0f));
    orients.append(qDegreesToRadians(90.0f));
    worker->createOpticFlowEstimator(fsettings,orients);

    connect(&updateTimer,SIGNAL(timeout()),this,SLOT(OnUpdate()));
    connect(this,SIGNAL(startProcessing()),worker,SLOT(start()));
    connect(&dvsEventHandler,SIGNAL(OnPlaybackFinished()),this,SLOT(OnPlaybackFinished()));

    emit startProcessing();

    //dvsEventHandler.PlayBackFile("/tausch/BottiBot/dvs128_towers_take_1_2016-12-22.aedat");
    dvsEventHandler.PlayBackFile("/tausch/BottiBot/dvs128_wall_take_2_2016-12-22.aedat");
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
        Buffer2D oppMoEnergy1,oppMoEnergy2;
        long time = worker->getMotionEnergy(0,0,oppMoEnergy1);
        worker->getMotionEnergy(0,1,oppMoEnergy2);

        if(time != -1){
            QImage img1 = oppMoEnergy1.toImage(-0.5,0.5);
            ui->label_1->setPixmap(QPixmap::fromImage(img1));
            QImage img2 = oppMoEnergy2.toImage(-0.5,0.5);
            ui->label_2->setPixmap(QPixmap::fromImage(img2));

            ui->l_timestamp->setText(QString("%1").arg(time));
        }else{
            qDebug("No new data available!");
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

        float p = worker->getProcessingRatio();
        ui->l_proc_ratio->setText(QString("%1 %").arg(p*100,0,'g',4));
    }
}
void MainWindow::OnNewEvent(DVSEventHandler::DVSEvent e)
{
    worker->setNextEvent(e);
}
void MainWindow::OnPlaybackFinished()
{
    worker->stopProcessing();

//    FilterSettings fset = FilterSettings::getSettings(FilterSettings::SPEED_25);

//    QList<float> orients;
//    QList<FilterSettings> fsettings;
//    fsettings.append(fset);
//    orients.append(qDegreesToRadians(0.0f));
//    //orients.append(qDegreesToRadians(90.0f));
//    worker->createOpticFlowEstimator(fsettings,orients);

//    emit startProcessing();
//    dvsEventHandler.PlayBackFile("/tausch/BottiBot/dvs128_towers_take_1_2016-12-22.aedat",0);
}
