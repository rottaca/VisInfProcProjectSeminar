#include "filtersettings.h"
#include <QtMath>

#include <assert.h>

FilterSettings::FilterSettings(float _f0, float _muBi1, float _tempEnd, float _tempSteps, int _spatialSz)
    :f0(_f0),
     s1(1.f/2),s2(3.f/4),
     muBi1(_muBi1),muBi2(2*muBi1),
     sigmaBi1(muBi1/3),sigmaBi2(3.f/2*sigmaBi1),
     muMono(1.f/5*muBi1*(1 + qSqrt(36 + 10*qLn(s1/s2)))),
     sigmaMono(muMono/3.f),
     sigmaGabor(25),
     temporalEnd(_tempEnd),
     temporalSteps(_tempSteps),
     spatialSize(_spatialSz)
{
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
                   "temporalEnd: %10\n"
                   "temporalSteps: %10\n"
                   "spatialSize: %10\n")
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
