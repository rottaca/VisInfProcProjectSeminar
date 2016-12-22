#include "mainwindow.h"
#include "ui_mainwindow.h"


#include <QtMath>
#include <QList>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    fsettings = FilterSettings::getSettings(FilterSettings::SPEED_25);

    QList<float> orients;
    orients.append(qDegreesToRadians(0.0f));
    opticFlowEstim = new MotionEnergyEstimator(fsettings,orients);

    qRegisterMetaType<DVSEventHandler::DVSEvent>("DVSEventHandler::DVSEvent");
    connect(&dvsEventHandler,SIGNAL(OnNewEvent(DVSEventHandler::DVSEvent)),opticFlowEstim,SLOT(OnNewEvent(DVSEventHandler::DVSEvent)));
    connect(ui->verticalSlider,SIGNAL(valueChanged(int)),this,SLOT(OnChangeSlider(int)));
    connect(opticFlowEstim,SIGNAL(ImageReady(QImage,QImage)),this,SLOT(OnImageReady(QImage,QImage)));

    OnChangeSlider(0);

    dvsEventHandler.PlayBackFile("/tausch/BottiBot/dvs128_corridor_take_1_2016-12-22.aedat",1);
}

void MainWindow::OnChangeSlider(int pos)
{
//    QImage img = fset.spatialTemporal[FilterSet::LEFT1].toImageXZ(pos).scaled(400, 400, Qt::KeepAspectRatio);
//    ui->label->setPixmap(QPixmap::fromImage(img));

}
void MainWindow::OnNewEvent(DVSEventHandler::DVSEvent e)
{
//    conv.convolute3D(fset.spatialTemporal[FilterSet::LEFT1],QVector2D(e.posX,e.posY));

//    QImage img = conv.getBuff()->toImageXZ(64).scaled(400, 400, Qt::KeepAspectRatio);
//    ui->label_2->setPixmap(QPixmap::fromImage(img));
//    Buffer2D b;
//    conv.nextTimeSlot(&b);
//    img = b.toImage().scaled(400, 400, Qt::KeepAspectRatio);
//    ui->label_3->setPixmap(QPixmap::fromImage(img));
//    Buffer2D b2(128,128);
//    b2(e.posX,e.posY) = 1;
//    img = b2.toImage().scaled(400, 400, Qt::KeepAspectRatio);
//    ui->label_4->setPixmap(QPixmap::fromImage(img));
}

void MainWindow::OnImageReady(QImage energy,QImage bufferXZ)
{
    energy = energy.scaled(400, 400);
    ui->label_4->setPixmap(QPixmap::fromImage(energy));
    bufferXZ = bufferXZ.scaled(400, 400, Qt::KeepAspectRatio);
    ui->label_3->setPixmap(QPixmap::fromImage(bufferXZ));
}

MainWindow::~MainWindow()
{
    delete ui;
}
