#ifndef FILTERSETTINGS_H
#define FILTERSETTINGS_H

#include <QString>

class FilterSettings
{
public:
    /**
     * @brief FilterSettings Computes the spatial temporal filter settings according to
     * psychovisual experiments.
     * @param _f0
     * @param _muBi1
     * @param _tempEnd End of temporal function
     * @param _tempSteps Number of steps to go from t = 0 to t = tempEnd
     * @param _spatialSz Size of spatial filter, has to be odd
     */
    FilterSettings(float _f0, float _muBi1, float _tempEnd, float _tempSteps, int _spatialSz);

    /**
     * @brief toString Converts the filtersettings into a string
     * @return
     */
    QString toString();

public:
    const float f0;
    const float s1,s2;
    const float muBi1,muBi2;
    const float sigmaBi1, sigmaBi2;
    const float muMono;
    const float sigmaMono;
    const float sigmaGabor;
    const float temporalEnd;
    const float temporalSteps;
    const int spatialSize;

};

#endif // FILTERSETTINGS_H
