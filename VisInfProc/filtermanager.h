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

    /**
     * @brief constructTemporalFilter Create the temporal bi or mono phasic filter
     * @param s
     * @param type
     * @return
     */
    static Buffer1D constructTemporalFilter(FilterSettings& s, enum TemporalFilter type);
    /**
     * @brief constructSpatialFilter Creates the spatial odd or even gabor filter
     * @param s
     * @param orientation
     * @param type
     * @return
     */
    static Buffer2D constructSpatialFilter(FilterSettings& s, float orientation, enum SpatialFilter type);
    /**
     * @brief combineFilters Combines a temporal 1D and a spatial 2D filter into a spatial temporal 3D filter
     * @param temporal
     * @param spatial
     * @return
     */
    static Buffer3D combineFilters(Buffer1D &temporal, Buffer2D &spatial);


private:
    static inline float gaussTemporal(float sigma, float mu, float t);
    static inline float gaussSpatial(float sigma, float x, float y);
};

#endif // FILTERMANAGER_H
