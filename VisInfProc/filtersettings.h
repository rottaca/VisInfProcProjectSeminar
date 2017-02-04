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
     * @param _alphaPNorm Normalization parameter
     * @param _alphaQNorm Normalization parameter
     * @param _betaNorm Normalization parameter
     * @param _sigmaNorm Normalization parameter
     * @param _speed_px_per_sec Prefered motion speed
     */
    FilterSettings(float _f0, float _muBi1, float _tempEnd,
                   float _tempSteps, int _spatialSz, int _timewindow_us,
                   float _alphaPNorm, float _alphaQNorm, float _betaNorm, float _sigmaNorm,
                   float _speed_px_per_sec);

    /**
     * @brief toString Converts the filtersettings into a string
     * @return
     */
    QString toString();

    enum PredefinedSettings {SPEED_1,SPEED_2,SPEED_3,SPEED_4,SPEED_5};
    static FilterSettings getSettings(enum PredefinedSettings ps);

public:
    // Spatial-temporal filter settings
    float f0;
    float s1,s2;
    float muBi1,muBi2;
    float sigmaBi1, sigmaBi2;
    float muMono;
    float sigmaMono;
    float sigmaGabor;
    // Filter resolution and time scale
    float temporalEnd;
    float temporalSteps;
    int timewindow_us;
    int spatialSize;
    // Normalization settings
    float alphaPNorm;
    float alphaQNorm;
    float betaNorm;
    float sigmaNorm;
    // Flow computation
    float speed_px_per_sec;

};

#endif // FILTERSETTINGS_H
