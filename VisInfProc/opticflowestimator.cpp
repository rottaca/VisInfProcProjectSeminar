#include "opticflowestimator.h"

OpticFlowEstimator::OpticFlowEstimator()
{
    currentStreamTime = 0;
    fset = NULL;
    conv = NULL;
}


OpticFlowEstimator::OpticFlowEstimator(FilterSettings fs, QList<float> orientations)
{
    this->fsettings = fs;
    this->orientations = orientations;
    currentStreamTime = 0;

    fset = new FilterSet[orientations.length()];
    conv = new Convolution3D[orientations.length()*4];
    for(int i = 0; i < orientations.length(); i++){
        fset[i] = FilterSet(fs,orientations.at(i));
        conv[i*4+0] = Convolution3D(128,128,
                  fset[i].spatialTemporal[FilterSet::LEFT1].getSizeZ());
        conv[i*4+1] = Convolution3D(128,128,
                  fset[i].spatialTemporal[FilterSet::LEFT1].getSizeZ());
        conv[i*4+2] = Convolution3D(128,128,
                  fset[i].spatialTemporal[FilterSet::LEFT1].getSizeZ());
        conv[i*4+3] = Convolution3D(128,128,
                  fset[i].spatialTemporal[FilterSet::LEFT1].getSizeZ());
    }
}
OpticFlowEstimator &OpticFlowEstimator::operator=(const OpticFlowEstimator& other)
{
    //TODO
    *this;
}

OpticFlowEstimator::~OpticFlowEstimator()
{
    if(fset != NULL)
        delete[] fset;
    fset = NULL;
    if(conv != NULL)
        delete[] fset;
    conv = NULL;
}
void OpticFlowEstimator::onNewEvent(DVSEventHandler::DVSEvent e)
{
    //TODO
}
