#include "filterset.h"
#include "filtermanager.h"

FilterSet::FilterSet(FilterSettings fs, float orientation)
{
    this->orientation = orientation;
    this->fs = fs;

    FilterManager fm;
    tempMono = fm.constructTemporalFilter(fs,FilterManager::MONO);
    tempBi = fm.constructTemporalFilter(fs,FilterManager::BI);

    gaborEven = fm.constructSpatialFilter(fs,orientation,FilterManager::EVEN);
    gaborOdd = fm.constructSpatialFilter(fs,orientation,FilterManager::ODD);

    spatialTemporal[EVEN_MONO] = fm.combineFilters(tempMono,gaborEven);
    spatialTemporal[EVEN_BI] = fm.combineFilters(tempBi,gaborEven);
    spatialTemporal[ODD_MONO] = fm.combineFilters(tempMono,gaborOdd);
    spatialTemporal[ODD_BI] = fm.combineFilters(tempBi,gaborOdd);
    spatialTemporal[LEFT1] = spatialTemporal[ODD_BI] - spatialTemporal[EVEN_MONO];
    spatialTemporal[LEFT2] = spatialTemporal[ODD_MONO] + spatialTemporal[EVEN_BI];
    spatialTemporal[RIGHT1] = spatialTemporal[ODD_MONO] - spatialTemporal[EVEN_BI];
    spatialTemporal[RIGHT2] = spatialTemporal[ODD_BI] + spatialTemporal[EVEN_MONO];

}
