#include "filtersettings.h"
#include <QtMath>

#include <assert.h>

FilterSettings::FilterSettings()
{
    f0 = 0;
    s1 = 0;
    s2 = 0;
    muBi1 = 0;
    muBi2 = 0;
    sigmaBi1 = 0;
    sigmaBi2 = 0;
    muMono = 0;
    sigmaMono = 0;
    sigmaGabor = 0;
    temporalEnd = 0;
    temporalSteps = 0;
    spatialSize = 0;
    timewindow_us = 0;
    alphaPNorm = 0;
    alphaQNorm = 0;
    betaNorm = 0;
    sigmaNorm = 0;
    speed_px_per_sec = 0;
}

FilterSettings::FilterSettings(float _f0, float _muBi1, float _tempEnd,
                               float _tempSteps, int _spatialSz, int _timewindow_us,
                               float _alphaPNorm, float _alphaQNorm, float _betaNorm, float _sigmaNorm,
                               float _speed_px_per_sec)
{
    f0 = _f0;
    s1 = 1.f/2;
    s2 = 3.f/4;
    muBi1 = _muBi1;
    muBi2 = 2*muBi1;
    sigmaBi1 = muBi1/3;
    sigmaBi2 = 3.f/2*sigmaBi1;
    muMono = 1.f/5*muBi1*(1 + qSqrt(36 + 10*qLn(s1/s2)));
    sigmaMono = muMono/3;
    sigmaGabor = 25.f;
    temporalEnd = _tempEnd;
    temporalSteps = _tempSteps;
    spatialSize = _spatialSz;
    timewindow_us = _timewindow_us;
    alphaPNorm = _alphaPNorm;
    alphaQNorm = _alphaQNorm;
    betaNorm = _betaNorm;
    sigmaNorm = _sigmaNorm;
    speed_px_per_sec = _speed_px_per_sec;

    assert(spatialSize % 2 != 0);
}

FilterSettings FilterSettings::getSettings(enum PredefinedSettings ps)
{
    switch (ps) {
    case DEFAULT:
            return FilterSettings(0.08f,0.2f,0.7,100,25,135000,0.1,0.002,1,3.6, 0);
        break;
    case SPEED_1:
            return FilterSettings(0.15f,0.23f,0.7,20,19,100000,0.1,0.002,1,3.6,66.733f);
        break;
    case SPEED_2:
            return FilterSettings(0.15f,0.23f,0.7,20,19,200000,0.1,0.002,1,3.6,33.366f);
        break;
    case SPEED_3:
            return FilterSettings(0.15f,0.23f,0.7,20,19,300000,0.1,0.002,1,3.6,22.244f);
        break;
    default:
            return FilterSettings();
        break;
    }
}
