#ifndef OPTICFLOWESTIMATOR_H
#define OPTICFLOWESTIMATOR_H

#include "buffer1d.h"
#include "buffer2d.h"
#include "buffer3d.h"
#include "filtermanager.h"
#include "filtersettings.h"
#include "filterset.h"
#include "convolution3d.h"
#include "dvseventhandler.h"

#include <QList>

class OpticFlowEstimator: public QObject
{
    Q_OBJECT
public:
    OpticFlowEstimator();
    OpticFlowEstimator(FilterSettings fs, QList<float> orientations);
    ~OpticFlowEstimator();

    OpticFlowEstimator &operator=(const OpticFlowEstimator& other);
public slots:
    void onNewEvent(DVSEventHandler::DVSEvent e);

private:
    FilterSettings fsettings;
    FilterSet* fset;
    Convolution3D* conv;
    QList<float> orientations;
    u_int32_t currentStreamTime;

};

#endif // OPTICFLOWESTIMATOR_H
