#include "filtermanager.h"
#include <QtMath>
#include "assert.h"

FilterManager::FilterManager()
{

}


Filter1D FilterManager::constructTemporalFilter(FilterSettings s, enum TemporalFilter type)
{
    Filter1D f1(s.temporalSteps);
    float tStep = s.temporalEnd/(s.temporalSteps-1);
    float t = 0;

    for(float i = 0; i < s.temporalSteps; i++){

        switch (type) {
        case BI:
                f1(i) = -s.s1*gaussTemporal(s.sigmaBi1,s.muBi1,t) +
                        s.s2*gaussTemporal(s.sigmaBi2,s.muBi2,t);
            break;
        case MONO:
                f1(i) = gaussTemporal(s.sigmaMono,s.muMono,t);
            break;
        }

        t+= tStep;
    }
    assert(qAbs(t-tStep-s.temporalEnd) < 1e-5);

    return f1;
}

Filter2D FilterManager::constructSpatialFilter(FilterSettings s, float orientation, enum SpatialFilter type)
{
    Filter2D f2(s.spatialSize,s.spatialSize);
    float fx0,fy0;
    fx0 = qCos(orientation)*s.f0;
    fy0 = qSin(orientation)*s.f0;

    int sz_2 = qFloor(s.spatialSize/2);
    for(int y = -sz_2; y <= sz_2; y++){
        for(int x = -sz_2; x <= sz_2; x++){
            // TODO Optimize
            float v =2*M_PI/(s.sigmaGabor*s.sigmaGabor)*gaussSpatial(s.sigmaGabor,x,y);

            float tmp = 2*M_PI*(fx0*x + fy0*y);

            switch (type) {
            case ODD:
                    v *= qSin(tmp);
                break;
            case EVEN:
                    v *= qCos(tmp);
                break;
            }

            f2(x+sz_2,y+sz_2) = v;
        }
    }

    return f2;
}

float FilterManager::gaussTemporal(float sigma, float mu, float t)
{
    return qExp(-(t-mu)*(t-mu)/(2*sigma*sigma));
}

float FilterManager::gaussSpatial(float sigma, float x, float y)
{
    return qExp(-2*M_PI*M_PI*(x*x+y*y)/(sigma*sigma));
}
