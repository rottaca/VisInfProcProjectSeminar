#ifndef FILTERMANAGER_H
#define FILTERMANAGER_H

#include "filtersettings.h"
#include "filter1d.h"
#include "filter2d.h"
#include "filter3d.h"

class FilterManager
{
public:

    enum TemporalFilter {BI,MONO};
    enum SpatialFilter {ODD, EVEN};

    FilterManager();

    Filter1D constructTemporalFilter(FilterSettings s, enum TemporalFilter type);
    Filter2D constructSpatialFilter(FilterSettings s, float orientation, enum SpatialFilter type);



private:
    inline float gaussTemporal(float sigma, float mu, float t);
    inline float gaussSpatial(float sigma, float x, float y);
};

#endif // FILTERMANAGER_H
