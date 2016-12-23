#include "filterset.h"
#include "filtermanager.h"
#include <iostream>

#include "cuda_helper.h"

FilterSet::FilterSet()
{
    sx = sy = sz = 0;

    for(int i = 0; i < CNT; i++){
        gpuSpatialTemporal[i] = NULL;
    }
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
    spatialTemporal[LEFT1] = spatialTemporal[ODD_BI] + spatialTemporal[EVEN_MONO];
    spatialTemporal[RIGHT1] = spatialTemporal[ODD_BI] - spatialTemporal[EVEN_MONO];
    spatialTemporal[RIGHT2] = spatialTemporal[ODD_MONO] + spatialTemporal[EVEN_BI];
    spatialTemporal[LEFT2] = spatialTemporal[ODD_MONO] - spatialTemporal[EVEN_BI];

    // Copy filters to GPU
    qDebug("Uploading filterset to GPU...");
    for(int i = 0; i < CNT; i++){
        long s = sx*sy*sz;
        gpuSpatialTemporal[i] = cudaCreateDoubleBuffer(s);
        cudaUploadDoubleBuffer(spatialTemporal[i].getBuff(),
                               gpuSpatialTemporal[i],
                               s);
    }
    qDebug("Filterset uploaded.");
}
FilterSet::~FilterSet(){

    for(int i = 0; i < CNT; i++){
        if(gpuSpatialTemporal[i] != NULL){
            cudaFree(gpuSpatialTemporal[i]);
        }
    }
}
