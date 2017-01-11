#include "filterset.h"
#include "filtermanager.h"
#include <iostream>

#include "cuda_helper.h"

#include <QImage>
#include <QFile>

FilterSet::FilterSet()
{
    sx = sy = sz = 0;
}

FilterSet::FilterSet(FilterSettings fs, float orientation)
{
    this->orientation = orientation;
    this->fs = fs;
    sz = fs.temporalSteps;
    sx = sy = fs.spatialSize;

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
    spatialTemporal[PHASE1] = spatialTemporal[ODD_BI];
    spatialTemporal[PHASE1] += spatialTemporal[EVEN_MONO];
    spatialTemporal[PHASE2] = spatialTemporal[ODD_MONO];
    spatialTemporal[PHASE2] -= spatialTemporal[EVEN_BI];

//    QFile file("even.png");
//    file.open(QIODevice::WriteOnly);
//    spatialTemporal[LEFT1].toImageXZ(12).save(&file,"PNG");
//    gaborOdd.toImage().save(&file,"PNG");

}
FilterSet::~FilterSet(){

}
