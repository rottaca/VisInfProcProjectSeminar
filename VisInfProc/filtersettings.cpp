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
    alphaPNorm = 0.1f;
    alphaQNorm = 0.002f;
    betaNorm = 1.0f;
    sigmaNorm = 3.6f;
    speed_px_per_sec = _speed_px_per_sec;

    assert(spatialSize % 2 != 0);
}

QString FilterSettings::toString()
{
    return QString("f0: %1\n"
                   "s1: %2\n"
                   "s2: %3\n"
                   "muBi1: %4\n"
                   "muBi2: %5\n"
                   "sigmaBi1: %6\n"
                   "sigmaBi2: %7\n"
                   "muMono: %8\n"
                   "sigmaMono: %9\n"
                   "sigmaGabor: %10\n"
                   "temporalEnd: %11\n"
                   "temporalSteps: %12\n"
                   "spatialSize: %13\n")
            .arg(f0)
            .arg(s1)
            .arg(s2)
            .arg(muBi1)
            .arg(muBi2)
            .arg(sigmaBi1)
            .arg(sigmaBi2)
            .arg(muMono)
            .arg(sigmaMono)
            .arg(sigmaGabor)
            .arg(temporalEnd)
            .arg(temporalSteps)
            .arg(spatialSize);
}

FilterSettings FilterSettings::getSettings(enum PredefinedSettings ps)
{
    switch (ps) {
    case DEFAULT:
            return FilterSettings(0.08f,0.2f,0.7,100,25,135000, 0);
        break;
    case SPEED_12_5:
            return FilterSettings(0.15f,0.23f,0.7,20,21,135000*2,12.5f);
        break;
    case SPEED_25:
            return FilterSettings(0.15f,0.23f,0.7,20,21,135000,25);
        break;
    case SPEED_50:
            return FilterSettings(0.15f,0.23f,0.7,20,21,135000/2,50);
        break;
    default:
            return FilterSettings(0,0,0,0,0,0,0);
        break;
    }
}
