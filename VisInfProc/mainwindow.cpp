#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "buffer1d.h"
#include "buffer2d.h"
#include "buffer3d.h"
#include "filtermanager.h"
#include "filtersettings.h"
#include "filterset.h"

#include <QtMath>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    FilterSettings fs(0.08f,0.2f,0.7,100,25);
    qDebug(QString("Filter: \n%1").arg(fs.toString()).toLocal8Bit());

    FilterSet fset(fs,0);

    QImage img = fset.spatialTemporal[FilterSet::LEFT2].toImageXZ(15).scaled(500, 500, Qt::KeepAspectRatio);
    ui->label->setPixmap(QPixmap::fromImage(img));
}

MainWindow::~MainWindow()
{
    delete ui;
}
