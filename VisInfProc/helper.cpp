#include "helper.h"

QColor Helper::pseudoColor(float v,float min, float max)
{
    QColor b(Qt::blue);
    QColor g(Qt::green);
    QColor r(Qt::red);

    if(v <= min)
        return b;
    else if(v >= max)
        return r;

    float range_2 = qAbs(max-min)/2;
    float mid = min + range_2;

    float t = qAbs(v-mid)/range_2;

    if(v < mid){
        return QColor(t*b.red() + (1-t)*g.red(),
                    t*b.green() + (1-t)*g.green(),
                    t*b.blue() + (1-t)*g.blue());
    }else{
        return QColor(t*r.red() + (1-t)*g.red(),
                    t*r.green() + (1-t)*g.green(),
                    t*r.blue() + (1-t)*g.blue());
    }
}
