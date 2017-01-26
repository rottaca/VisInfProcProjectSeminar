#include "pushbotcontroller.h"

#include "worker.h"
#include "edvsinterface.h"


#include <QDateTime>
#include <QtMath>

PushBotController::PushBotController(QObject* parent):QObject(parent)
{
    thread.start();
    moveToThread(&thread);
    processIntervalTimer.moveToThread(&thread);
    eventProcessor = NULL;
    robotInterface = NULL;

    P = 100;
    I = 0;
    D = 0;
    out = 0;
    eOld = 0;
    eSum = 0;
    eSumMax = 0;
    connect(&processIntervalTimer,SIGNAL(timeout()),this,SLOT(processFlow()));
}
PushBotController::~PushBotController()
{
    qDebug("Destroying pushBotController...");
    if(processIntervalTimer.isActive())
        processIntervalTimer.stop();

    thread.quit();
    if(!thread.wait(1000)){
        thread.terminate();
        thread.wait();
    }
}

void PushBotController::startProcessing(){
    processIntervalTimer.start(1000/PUSH_BOT_PROCESS_FPS);
}

void PushBotController::stopProcessing(){
    processIntervalTimer.stop();
}

void PushBotController::processFlow()
{
    QMutexLocker locker(&mutex);

    // Get all optic flows
    int speeds = settings.length();
    Buffer2D flowX[speeds], flowY[speeds];
    float* opticFlowPtrX[speeds];
    float* opticFlowPtrY[speeds];

    for(int i = 0; i < speeds; i++){
        eventProcessor->getOpticFlow(flowX[i],flowY[i],i);
        opticFlowPtrX[i] = flowX[i].getCPUPtr();
        opticFlowPtrY[i] = flowY[i].getCPUPtr();
    }
    int sx = flowX[0].getSizeX();
    int sy = flowX[0].getSizeY();
    flowXCombined.resize(sx,sy);
    flowYCombined.resize(sx,sy);

    float* combinedFlowXPtr = flowXCombined.getCPUPtr();
    float* combinedFlowYPtr = flowYCombined.getCPUPtr();

    float threshold = 0.3f;
    for(int i = 0; i < sx*sy; i++){
        float outX = 0;
        float outY = 0;
        float oldS = 0;
        int maxIdx = -1;
        for(int j = 0; j < speeds; j++){
            float fx,fy;
            fx = opticFlowPtrX[j][i];
            fy = opticFlowPtrY[j][i];
            float s = sqrt(fx*fx+fy*fy);
            if(s >= threshold && s > oldS){
                outX = fx/s*settings.at(j).speed_px_per_sec;
                outY = fy/s*settings.at(j).speed_px_per_sec;
                oldS = s;
                maxIdx = j;
            }
        }

        if(maxIdx >= 0){
            float x1,x2,x3,y1,y2,y3;
            float fx,fy;
            int idxCenter;
            // Find x and y coordinates for interpolation
            if(maxIdx == 0){
                x1 = settings.at(maxIdx).speed_px_per_sec;
                x2 = settings.at(maxIdx+1).speed_px_per_sec;
                x3 = settings.at(maxIdx+2).speed_px_per_sec;
                idxCenter = maxIdx+1;
                fx = opticFlowPtrX[maxIdx][i];
                fy = opticFlowPtrY[maxIdx][i];
                y1 = sqrt(fx*fx+fy*fy);
                fx = opticFlowPtrX[maxIdx+1][i];
                fy = opticFlowPtrY[maxIdx+1][i];
                y2 = sqrt(fx*fx+fy*fy);
                fx = opticFlowPtrX[maxIdx+2][i];
                fy = opticFlowPtrY[maxIdx+2][i];
                y3 = sqrt(fx*fx+fy*fy);
            }
            else if(maxIdx == speeds-1){
                x1 = settings.at(maxIdx-2).speed_px_per_sec;
                x2 = settings.at(maxIdx-1).speed_px_per_sec;
                x3 = settings.at(maxIdx).speed_px_per_sec;
                idxCenter = maxIdx-1;
                fx = opticFlowPtrX[maxIdx-2][i];
                fy = opticFlowPtrY[maxIdx-2][i];
                y1 = sqrt(fx*fx+fy*fy);
                fx = opticFlowPtrX[maxIdx-1][i];
                fy = opticFlowPtrY[maxIdx-1][i];
                y2 = sqrt(fx*fx+fy*fy);
                fx = opticFlowPtrX[maxIdx][i];
                fy = opticFlowPtrY[maxIdx][i];
                y3 = sqrt(fx*fx+fy*fy);
            }else{
                x1 = settings.at(maxIdx-1).speed_px_per_sec;
                x2 = settings.at(maxIdx).speed_px_per_sec;
                x3 = settings.at(maxIdx+1).speed_px_per_sec;
                idxCenter = maxIdx;
                fx = opticFlowPtrX[maxIdx-1][i];
                fy = opticFlowPtrY[maxIdx-1][i];
                y1 = sqrt(fx*fx+fy*fy);
                fx = opticFlowPtrX[maxIdx][i];
                fy = opticFlowPtrY[maxIdx][i];
                y2 = sqrt(fx*fx+fy*fy);
                fx = opticFlowPtrX[maxIdx+1][i];
                fy = opticFlowPtrY[maxIdx+1][i];
                y3 = sqrt(fx*fx+fy*fy);
            }

            float d2 = 2*((y3-y2)/(x3-x2)-(y1-y2)/(x1-x2))/(x3-x1);
            float d1 = 0;
            if ((x3+x1)>=(x2+x2))
                d1 = (y3-y2)/(x3-x2) - 0.5*d2*(x3-x2);
            else
                d1 = (y2-y1)/(x2-x1) + 0.5*d2*(x2-x1);
            if(d2 < 0){
                float xe = x2 - d1/d2;
                if(xe >= x1 && xe <= x3){
                    float ye = y2 + 0.5*d1*(xe-x2);
                    outX = 0;
                    outY = 0;
                    // ye is interpolated flow energy for speed xe
                    // Next step: Decompose in x and y direction
                    // Interpolate orientation linear between the left and right point from xe
                    if(xe < x2){
                        // Interpolation parameter for (x1,y1)
                        float t = (xe - x1)/(x2-x1);

                        float fx = opticFlowPtrX[idxCenter-1][i];
                        float fy = opticFlowPtrY[idxCenter-1][i];
                        float y = sqrt(fx*fx+fy*fy);
                        outX = fx/y*t;
                        outY = fy/y*t;
                        fx = opticFlowPtrX[idxCenter][i];
                        fy = opticFlowPtrY[idxCenter][i];
                        y = sqrt(fx*fx+fy*fy);
                        outX += fx/y*(1-t);
                        outY += fy/y*(1-t);
                        // Scale to new speed
                        outX *= xe;
                        outY *= xe;
                        //outX = 50*t;
                        //outY = 50*(1-t);
                    }else{
                        // Interpolation parameter for (x3,y3)
                        float t = (x3 - xe)/(x3-x2);

                        float fx = opticFlowPtrX[idxCenter+1][i];
                        float fy = opticFlowPtrY[idxCenter+1][i];
                        float y = sqrt(fx*fx+fy*fy);
                        outX = fx/y*t;
                        outY = fy/y*t;
                        fx = opticFlowPtrX[idxCenter][i];
                        fy = opticFlowPtrY[idxCenter][i];
                        y = sqrt(fx*fx+fy*fy);
                        outX += fx/y*(1-t);
                        outY += fy/y*(1-t);
                        // Scale to new speed
                        outX *= xe;
                        outY *= xe;
                        //outX = 50*t;
                        //outY = 50*(1-t);
                    }
                    // Interpolated speed and orientation: red
//                    outX = 50;
//                    outY = 0;
                }else{
                    // Maxima outside range: blue
//                    outX = -50;
//                    outY = 0;
//                    outX = 0;
//                    outY = 0;
                }
            }else{
                // Parabel has no maximum, use strongest response: green
//                outX = 0;
//                outY = 50;
//                outX = 0;
//                outY = 0;
            }
        }

        combinedFlowXPtr[i] = outX;
        combinedFlowYPtr[i] = outY;
    }

    // Compute average flow on left and right image half

    avgFlowVecXL = 0;
    avgFlowVecYL = 0;
    avgFlowVecXR = 0;
    avgFlowVecYR = 0;
    int cntL = 0,cntR = 0;

    for(int j = 0; j < sx*sy;j++){
        float fx = combinedFlowXPtr[j];
        float fy = combinedFlowYPtr[j];

        float s = qSqrt(fx*fx+fy*fy);
        if(s > 0){
            // Left or right image half
            if(j % sx < sx/2){
                avgFlowVecXL += fx;
                avgFlowVecYL += fy;
                cntL++;
            }else{
                avgFlowVecXR += fx;
                avgFlowVecYR += fy;
                cntR++;
            }
        }
    }

    // Normalize
    avgFlowVecXL /= cntL;
    avgFlowVecYL /= cntL;
    avgFlowVecXR /= cntR;
    avgFlowVecYR /= cntR;

//    qDebug("%f %f", sqrt(avgFlowVecXL*avgFlowVecXL+avgFlowVecYL*avgFlowVecYL),
//           sqrt(avgFlowVecXR*avgFlowVecXR+avgFlowVecYR*avgFlowVecYR));

    float deltaT;
    if(!loopTime.isValid())
        deltaT = 0;
    else{
        deltaT = loopTime.elapsed()/1000.0f;
    }
    loopTime.restart();

    {
        QMutexLocker locker(&pidMutex);
        // Simple PID-Controller
        float e = avgFlowVecXL-avgFlowVecXR;

        eSum = qMax(-eSumMax,qMin(eSum + e,eSumMax));
        out = P*e + I*deltaT*eSum + D/deltaT*(e-eOld);
        eOld = e;
    }
}
