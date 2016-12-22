#ifndef MOTIONENERGYESTIMATOR_H
#define MOTIONENERGYESTIMATOR_H

#include "buffer1d.h"
#include "buffer2d.h"
#include "buffer3d.h"
#include "filtermanager.h"
#include "filtersettings.h"
#include "filterset.h"
#include "convolution3d.h"
#include "dvseventhandler.h"

#include <vector>

class MotionEnergyEstimator: public QObject
{
    Q_OBJECT
public:
    MotionEnergyEstimator(FilterSettings fs, QList<float> orientations);
    ~MotionEnergyEstimator();

public slots:
    void OnNewEvent(DVSEventHandler::DVSEvent e);

signals:
    void ImageReady(QImage energy, QImage bufferXZ);

private:
    void computeMotionEnergy(Buffer2D &one, Buffer2D &two, Buffer2D &energy);

private:
    FilterSettings fsettings;
    FilterSet* fset;
    Convolution3D* conv;
    QList<float> orientations;
    float currentWindowStartTime;
    float timeRes;
    std::vector<DVSEventHandler::DVSEvent> events;

};

#endif // MOTIONENERGYESTIMATOR_H
