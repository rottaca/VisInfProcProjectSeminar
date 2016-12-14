#ifndef FILTERSET_H
#define FILTERSET_H

#include "buffer1d.h"
#include "buffer2d.h"
#include "buffer3d.h"
#include "filtersettings.h"
#include "filtermanager.h"

class FilterSet
{
public:
    FilterSet(FilterSettings fs, float orientation);

public:
    Buffer1D tempMono, tempBi;
    Buffer2D gaborOdd, gaborEven;
    enum FilterName{ODD_MONO,ODD_BI,EVEN_MONO,EVEN_BI,LEFT1,LEFT2,RIGHT1,RIGHT2,CNT};
    Buffer3D spatialTemporal[CNT];
    float orientation;
    FilterSettings fs;
};

#endif // FILTERSET_H
