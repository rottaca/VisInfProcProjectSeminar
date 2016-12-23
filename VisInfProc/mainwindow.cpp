#include "mainwindow.h"
#include "ui_mainwindow.h"


#include <QtMath>
#include <QList>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    worker = new Worker();
    FilterSettings fset = FilterSettings::getSettings(FilterSettings::SPEED_25);

    QList<float> orients;
    QList<FilterSettings> fsettings;
    fsettings.append(fset);
    orients.append(qDegreesToRadians(0.0f));
    //orients.append(qDegreesToRadians(90.0f));
    worker->createOpticFlowEstimator(fsettings,orients);

    qRegisterMetaType<DVSEventHandler::DVSEvent>("DVSEventHandler::DVSEvent");
    connect(&dvsEventHandler,SIGNAL(OnNewEvent(DVSEventHandler::DVSEvent)),this,SLOT(OnNewEvent(DVSEventHandler::DVSEvent)));
//    connect(ui->verticalSlider,SIGNAL(valueChanged(int)),this,SLOT(OnChangeSlider(int)));
//    connect(opticFlowEstim,SIGNAL(ImageReady(QImage,QImage)),this,SLOT(OnImageReady(QImage,QImage)));

    connect(&updateTimer,SIGNAL(timeout()),this,SLOT(OnUpdate()));
    connect(this,SIGNAL(startProcessing()),worker,SLOT(start()));
    connect(&dvsEventHandler,SIGNAL(OnPlaybackFinished()),this,SLOT(OnPlaybackFinished()));

    emit startProcessing();

    dvsEventHandler.PlayBackFile("/tausch/BottiBot/dvs128_wall_take_3_2016-12-22.aedat",0);
    //dvsEventHandler.PlayBackFile("/tausch/scale4/mnist_0_scale04_0550.aedat",0);

    updateTimer.start(1000/FPS);
}
MainWindow::~MainWindow()
{
    delete ui;
}
void MainWindow::OnUpdate()
{
    if(worker->getIsProcessing()){
        Buffer2D l1,r1,l2,r2;
        long time = worker->getMotionEnergy(0,0,l1,r1);
        //worker->getMotionEnergy(0,1,l2,r2);

        if(time != -1){
            QImage imgl1 = l1.toImage(0,0.5).scaled(400, 400);
            QImage imgr1 = r1.toImage(0,0.5).scaled(400, 400);
            //QImage imgl2 = l2.toImage(0,0.5).scaled(400, 400);
            //QImage imgr2 = r2.toImage(0,0.5).scaled(400, 400);
            ui->label_1->setPixmap(QPixmap::fromImage(imgl1));
            ui->label_2->setPixmap(QPixmap::fromImage(imgr1));
            //ui->label_3->setPixmap(QPixmap::fromImage(imgl2));
            //ui->label_4->setPixmap(QPixmap::fromImage(imgr2));
        }
        float p = worker->getProcessingRatio();
        qDebug(QString("Ratio: %1").arg(p).toLocal8Bit());
    }
}
void MainWindow::OnNewEvent(DVSEventHandler::DVSEvent e)
{
    worker->setNextEvent(e);
}
void MainWindow::OnPlaybackFinished()
{
    worker->stopProcessing();
}

