#include "mainwindow.h"
#include "ui_mainwindow.h"


#include <QtMath>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    fsettings = FilterSettings::getSettings(FilterSettings::SPEED_25);
    fset = FilterSet(fsettings,qDegreesToRadians(45.0f));
    conv = Convolution3D(128,
                         128,
                         fset.spatialTemporal[FilterSet::LEFT1].getSizeZ());

    dvsEventHandler.playBackFile("/tausch/scale4/mnist_0_scale04_0550.aedat",1);

    qRegisterMetaType<DVSEventHandler::DVSEvent>("DVSEventHandler::DVSEvent");
    connect(&dvsEventHandler,SIGNAL(OnNewEvent(DVSEventHandler::DVSEvent)),this,SLOT(OnNewEvent(DVSEventHandler::DVSEvent)));
    connect(ui->verticalSlider,SIGNAL(valueChanged(int)),this,SLOT(OnChangeSlider(int)));
    OnChangeSlider(0);
}

void MainWindow::OnChangeSlider(int pos)
{
    QImage img = fset.spatialTemporal[FilterSet::LEFT1].toImageXZ(pos).scaled(400, 400, Qt::KeepAspectRatio);
    ui->label->setPixmap(QPixmap::fromImage(img));

}
void MainWindow::OnNewEvent(DVSEventHandler::DVSEvent e)
{
    conv.convolute3D(fset.spatialTemporal[FilterSet::LEFT1],QVector2D(e.posX,e.posY));

    QImage img = conv.getBuff()->toImageXZ(64).scaled(400, 400, Qt::KeepAspectRatio);
    ui->label_2->setPixmap(QPixmap::fromImage(img));
    Buffer2D b;
    conv.nextTimeSlot(&b);
    img = b.toImage().scaled(400, 400, Qt::KeepAspectRatio);
    ui->label_3->setPixmap(QPixmap::fromImage(img));
    Buffer2D b2(128,128);
    b2(e.posX,e.posY) = 1;
    img = b2.toImage().scaled(400, 400, Qt::KeepAspectRatio);
    ui->label_4->setPixmap(QPixmap::fromImage(img));
}

MainWindow::~MainWindow()
{
    delete ui;
}
