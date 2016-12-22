#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include "buffer1d.h"
#include "buffer2d.h"
#include "buffer3d.h"
#include "filtermanager.h"
#include "filtersettings.h"
#include "filterset.h"
#include "convolution3d.h"
#include "dvseventhandler.h"
#include "motionenergyestimator.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

public slots:
    void OnChangeSlider(int pos);
    void OnNewEvent(DVSEventHandler::DVSEvent e);
    void OnImageReady(QImage energy, QImage bufferXZ);

private:
    Ui::MainWindow *ui;
    FilterSettings fsettings;
    DVSEventHandler dvsEventHandler;
    MotionEnergyEstimator* opticFlowEstim;
};

#endif // MAINWINDOW_H
