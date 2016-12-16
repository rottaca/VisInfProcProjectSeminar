#include "filterset.h"
#include "filtermanager.h"

FilterSet::FilterSet(FilterSettings fs, float orientation)
{
    this->orientation = orientation;
    this->fs = fs;

    // Construct temporal filters
    tempMono = FilterManager::constructTemporalFilter(fs,FilterManager::MONO);
    tempBi = FilterManager::constructTemporalFilter(fs,FilterManager::BI);
    // Construct spatial filters
    gaborEven = FilterManager::constructSpatialFilter(fs,orientation,FilterManager::EVEN);
    gaborOdd = FilterManager::constructSpatialFilter(fs,orientation,FilterManager::ODD);

    // Construct spatial temporal filters
    spatialTemporal[EVEN_MONO] = FilterManager::combineFilters(tempMono,gaborEven);
    spatialTemporal[EVEN_BI] = FilterManager::combineFilters(tempBi,gaborEven);
    spatialTemporal[ODD_MONO] = FilterManager::combineFilters(tempMono,gaborOdd);
    spatialTemporal[ODD_BI] = FilterManager::combineFilters(tempBi,gaborOdd);
    // Construct differences
    spatialTemporal[LEFT1] = spatialTemporal[ODD_BI] - spatialTemporal[EVEN_MONO];
    spatialTemporal[RIGHT1] = spatialTemporal[ODD_BI] + spatialTemporal[EVEN_MONO];
    spatialTemporal[RIGHT2] = spatialTemporal[ODD_MONO] - spatialTemporal[EVEN_BI];
    spatialTemporal[LEFT2] = spatialTemporal[ODD_MONO] + spatialTemporal[EVEN_BI];

}
