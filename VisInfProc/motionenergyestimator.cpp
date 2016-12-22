#include "motionenergyestimator.h"
#include <QtMath>
#include <QElapsedTimer>
#include <QPainter>
#include <assert.h>
#include <iostream>

MotionEnergyEstimator::MotionEnergyEstimator(FilterSettings fs, QList<float> orientations)
{
    this->fsettings = fs;
    this->orientations = orientations;
    currentWindowStartTime = 0;

    fset = new FilterSet[orientations.length()];
    conv = new Convolution3D[orientations.length()*4];
    for(int i = 0; i < orientations.length(); i++){
        fset[i] = FilterSet(fs,orientations.at(i));
        for(int j = 0; j < 4; j++){
            conv[i*4+j] = Convolution3D(128,128,fset[i].spatialTemporal[FilterSet::LEFT1].getSizeZ());
        }
    }
    // Time per timeslot
    timeRes =(float)fs.timewindow_us/fs.temporalSteps;
}

MotionEnergyEstimator::~MotionEnergyEstimator()
{
    if(fset != NULL)
        delete[] fset;
    fset = NULL;
    if(conv != NULL)
        delete[] fset;
    conv = NULL;
}
void MotionEnergyEstimator::OnNewEvent(DVSEventHandler::DVSEvent e)
{
    if(e.On)
        return;

    //if(currentWindowStartTime > 0)
    //    return;

    // Events are sorted by time
    assert(events.size() == 0 || e.timestamp >= events.back().timestamp);

    events.push_back(e);
    QVector2D ePos(e.posX,e.posY);
    //QVector2D ePos(63,63);

    int deltaT = e.timestamp - currentWindowStartTime;
    // Do we have to skip any timeslots ? Is the new event too new for the current slot ?
    int timeSlotsToSkip = qFloor((float)deltaT/timeRes);
    //qDebug(QString("%1").arg((float)deltaT/timeRes).toLocal8Bit());
    // Skip the slots and convolute
    QElapsedTimer t;
    t.start();
    for(int i = 0; i < orientations.length(); i++){

        // Skip time slots
        // TODO Skip everything ?
        //qDebug("ProcessEvent");
        if(timeSlotsToSkip > 0){
            Buffer2D left1,left2,right1,right2;
            conv[i*4+0].nextTimeSlot(&left1,timeSlotsToSkip);
            conv[i*4+1].nextTimeSlot(&left2,timeSlotsToSkip);
            conv[i*4+2].nextTimeSlot(&right1,timeSlotsToSkip);
            conv[i*4+3].nextTimeSlot(&right2,timeSlotsToSkip);
            Buffer2D motionEnergyLeft, motionEnergyRight;
            computeMotionEnergy(left1,left2,motionEnergyLeft);
            computeMotionEnergy(right1,right2,motionEnergyRight);

            double* ptrOne = motionEnergyLeft.getBuff();
            double max = *std::max_element(ptrOne,ptrOne+128*128);

            qDebug(QString("%1").arg(max).toLocal8Bit());
            //emit ImageReady(motionEnergyLeft.toImage(0,0.5f),conv[i*4].getBuff()->toImageXZ(64,-0.2,0.2));
            QImage img = motionEnergyLeft.toImage(0,1.5f);

            for(int k = 0; k < events.size(); k++)
                img.setPixel(events.at(k).posX,events.at(k).posY,qRgb(0, 0, 0));
         //   emit ImageReady(img,
         //           conv[0].toOrderedImageXZ(conv[0].getWriteIdx(),13,-0.02,0.02));

            //emit ImageReady(fset[0].spatialTemporal[FilterSet::LEFT1].toImageXZ(13),
            //    conv[0].toOrderedImageXZ(conv[0].getWriteIdx(),13,-0.015,0.015));

            emit ImageReady(img,
                    conv[0].toOrderedImageXZ(conv[1].getWriteIdx(),63,-0.2,0.2));
//            std::cout << "Image" << std::endl;
//                for(int y = 0; y < 100; y++){
//                    for(int x = 0; x < 127; x++){
//                        std::cout << conv[0].getBuff()->operator ()(x,63,y) << ", ";
//                    }
//                    std::cout << conv[0].getBuff()->operator ()(127,63,y) << std::endl;
//                }

        }
        // Convolute all four filters for this direction
        conv[i*4+0].convolute3D(fset[i].spatialTemporal[FilterSet::LEFT1],ePos);
        conv[i*4+1].convolute3D(fset[i].spatialTemporal[FilterSet::LEFT2],ePos);
        conv[i*4+2].convolute3D(fset[i].spatialTemporal[FilterSet::RIGHT1],ePos);
        conv[i*4+3].convolute3D(fset[i].spatialTemporal[FilterSet::RIGHT2],ePos);
    }

    //currentWindowStartTime++;

    qint64 nanoSec = t.nsecsElapsed();
    if(timeSlotsToSkip > 0){
        currentWindowStartTime += timeRes*timeSlotsToSkip;
        while(!events.empty() &&
              (int64_t)events.front().timestamp < (int64_t)currentWindowStartTime - fsettings.timewindow_us)
            events.erase(events.begin());
    }

    //if(e.timestamp == 0)
    //qInfo(QString("All convolutions done in: %1 microSec").arg(nanoSec/1000).toLocal8Bit());
    //qInfo(QString("Time: %1 microSec").arg(e.timestamp).toLocal8Bit());
}

void MotionEnergyEstimator::computeMotionEnergy(Buffer2D &one, Buffer2D &two, Buffer2D &energy)
{
    double* ptrOne = one.getBuff();
    double* ptrTwo = two.getBuff();
    energy.resize(one.getSizeX(),one.getSizeY());
    double* ptrEnergy = energy.getBuff();

    for(int i = 0; i < one.getSizeX()*one.getSizeY(); i++){
        ptrEnergy[i] = qSqrt(ptrOne[i]*ptrOne[i] + ptrTwo[i]*ptrTwo[i]);
    }
}
