#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "filter1d.h"
#include "filter2d.h"
#include "filter3d.h"
#include "filtermanager.h"
#include "filtersettings.h"
#include <QtMath>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    FilterSettings fs(0.08f,0.2f,0.7,100,31);
    FilterManager fm;
    Filter1D f1 = fm.constructTemporalFilter(fs,FilterManager::MONO);
    Filter2D f2 = fm.constructSpatialFilter(fs,qDegreesToRadians(0.f),FilterManager::EVEN);

    QImage img = f2.toImage();
    ui->label->setPixmap(QPixmap::fromImage(img));
}

MainWindow::~MainWindow()
{
    delete ui;
}
