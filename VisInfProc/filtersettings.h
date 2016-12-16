#ifndef FILTERSETTINGS_H
#define FILTERSETTINGS_H

#include <QString>

class FilterSettings
{
public:

    FilterSettings();
    /**
     * @brief FilterSettings Computes the spatial temporal filter settings according to
     * psychovisual experiments.
     * @param _f0
     * @param _muBi1
     * @param _tempEnd End of temporal function
     * @param _tempSteps Number of steps to go from t = 0 to t = tempEnd
     * @param _spatialSz Size of spatial filter, has to be odd
     * @param _timewindow timewindow for filter
     */
    FilterSettings(float _f0, float _muBi1, float _tempEnd, float _tempSteps, int _spatialSz, int _timewindow_us);

    /**
     * @brief toString Converts the filtersettings into a string
     * @return
     */
    QString toString();

    enum PredefinedSettings {DEFAULT,SPEED_25};
    static FilterSettings getSettings(enum PredefinedSettings ps);

public:
    float f0;
    float s1,s2;
    float muBi1,muBi2;
    float sigmaBi1, sigmaBi2;
    float muMono;
    float sigmaMono;
    float sigmaGabor;
    float temporalEnd;
    float temporalSteps;
    int spatialSize;
    int timewindow_us;

};

#endif // FILTERSETTINGS_H
