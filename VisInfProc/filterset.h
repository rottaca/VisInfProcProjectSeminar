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
    FilterSet();
    FilterSet(FilterSettings fs, float orientation);
    ~FilterSet();
public:
    Buffer1D tempMono, tempBi;
    Buffer2D gaborOdd, gaborEven;
    enum FilterName {ODD_MONO,ODD_BI,EVEN_MONO,EVEN_BI,PHASE1,PHASE2,CNT};
    Buffer3D spatialTemporal[CNT];
    float orientation;
    FilterSettings fs;

    int sx,sy,sz;
};

#endif // FILTERSET_H
