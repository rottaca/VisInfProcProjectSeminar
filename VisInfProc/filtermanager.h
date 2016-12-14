#ifndef FILTERMANAGER_H
#define FILTERMANAGER_H

#include "filtersettings.h"
#include "buffer1d.h"
#include "buffer2d.h"
#include "buffer3d.h"

class FilterManager
{
public:

    enum TemporalFilter {BI,MONO};
    enum SpatialFilter {ODD, EVEN};

    FilterManager();

    Buffer1D constructTemporalFilter(FilterSettings s, enum TemporalFilter type);
    Buffer2D constructSpatialFilter(FilterSettings s, float orientation, enum SpatialFilter type);

    Buffer3D combineFilters(Buffer1D temporal, Buffer2D spatial);


private:
    inline float gaussTemporal(float sigma, float mu, float t);
    inline float gaussSpatial(float sigma, float x, float y);
};

#endif // FILTERMANAGER_H
