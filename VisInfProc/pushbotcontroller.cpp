#include "pushbotcontroller.h"

#include "convolutionHandler.h"
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

    Kp = PUSHBOT_PID_P_DEFAULT;
    Ki = PUSHBOT_PID_I_DEFAULT;
    Kd = PUSHBOT_PID_D_DEFAULT;
    out = 0;
    eOld = 0;
    eSum = 0;
    avgFlowValid = false;

    cudaStreamCreate(&cudaStream);
    opticFlowDir.setCudaStream(cudaStream);
    opticFlowSpeed.setCudaStream(cudaStream);
    opticFlowEnergy.setCudaStream(cudaStream);

    connect(&processIntervalTimer,SIGNAL(timeout()),this,SLOT(processFlow()));
    connect(this,SIGNAL(stopTimer()),&processIntervalTimer,SLOT(stop()));
}
PushBotController::~PushBotController()
{
    PRINT_DEBUG("Destroying pushBotController...");
    if(processIntervalTimer.isActive()) {
        emit stopTimer();
        stopProcessing();
    }

    thread.quit();
    if(!thread.wait(THREAD_WAIT_TIME_MS)) {
        qCritical("Failed to stop PushBotController!");
        thread.terminate();
        thread.wait();
    }
    cudaStreamDestroy(cudaStream);
}
void PushBotController::setRobotInterface(eDVSInterface* interface)
{
    QMutexLocker locker(&mutex);
    robotInterface = interface;
    connect(this,SIGNAL(setMotorVelocity(int,int)),robotInterface,SLOT(setMotorVelocity(int,int)));
    connect(this,SIGNAL(enableMotors(bool)),robotInterface,SLOT(enableMotors(bool)));
}
void PushBotController::startProcessing()
{
    PRINT_DEBUG("Start pushbot controller");
    processIntervalTimer.start(1000/PUSH_BOT_PROCESS_FPS);
    loopTime.invalidate();
    emit setMotorVelocity(0,PUSHBOT_VELOCITY_DEFAULT);
    emit setMotorVelocity(1,PUSHBOT_VELOCITY_DEFAULT);
    emit enableMotors(true);
    out = 0;
    eOld = 0;
    eSum = 0;
    avgFlowValid = false;
}

void PushBotController::stopProcessing()
{
    PRINT_DEBUG("Stop pushbot controller");
    processIntervalTimer.stop();
    emit setMotorVelocity(1,0);
    emit setMotorVelocity(0,0);
    emit enableMotors(false);
}

void PushBotController::processFlow()
{
    QMutexLocker locker(&mutex);

    eventProcessor->getOpticFlow(opticFlowSpeed,opticFlowDir,opticFlowEnergy);
    float* flowSpeedPtr = opticFlowSpeed.getCPUPtr();
    float* flowDirPtr = opticFlowDir.getCPUPtr();
    float* flowEnergyPtr = opticFlowEnergy.getCPUPtr();

    int sx = opticFlowSpeed.getSizeX();
    int sy = opticFlowSpeed.getSizeY();

    // Compute average flow on left and right image half
    // Weighted by their normalized energy (to ignore values with low propability)
    avgFlowVecXL = 0;
    avgFlowVecYL = 0;
    avgFlowVecXR = 0;
    avgFlowVecYR = 0;
    float energyL = 0,energyR = 0;
    // Horizontal component of motion energy
    float energyXL = 0, energyXR = 0;

    for(int j = 0; j < sx*sy; j++) {
        float dir = flowDirPtr[j];
        float s = flowSpeedPtr[j];
        float e = flowEnergyPtr[j];
        if(s > 0) {
            // Left or right image half
            if(j % sx < sx/2) {
                avgFlowVecXL += cos(dir)*s*e;
                avgFlowVecYL += sin(dir)*s*e;
                energyL+=e;
                energyXL+=cos(dir)*e;
            } else {
                avgFlowVecXR += cos(dir)*s*e;
                avgFlowVecYR += sin(dir)*s*e;
                energyR+=e;
                energyXR+=cos(dir)*e;
            }
        }
    }

    // Normalize
    bool leftFlowValid = energyL > PUSHBOT_MIN_DETECTION_ENERGY;
    bool rightFlowValid = energyR > PUSHBOT_MIN_DETECTION_ENERGY;
    avgFlowValid = leftFlowValid || rightFlowValid;

    if(leftFlowValid) {
        avgFlowVecXL /= energyL;
        avgFlowVecYL /= energyL;
    } else {
        avgFlowVecXL = 0;
        avgFlowVecYL = 0;
        energyXL = 0;
    }
    if(rightFlowValid) {
        avgFlowVecXR /= energyR;
        avgFlowVecYR /= energyR;
    } else {
        avgFlowVecXR = 0;
        avgFlowVecYR = 0;
        energyXR = 0;
    }
    PRINT_DEBUG_FMT("AvgSpeedR: %f",avgFlowVecXR);
    PRINT_DEBUG_FMT("Horizontal Energy: %f",energyXR);
    PRINT_DEBUG_FMT("AvgSpeedL: %f",avgFlowVecXL);
    PRINT_DEBUG_FMT("Horizontal Energy: %f",energyXL);

    // Compute output
    if(leftFlowValid || rightFlowValid) {

        // Get elapsed time
        float deltaT;
        if(!loopTime.isValid())
            deltaT = 0;
        else {
            deltaT = loopTime.nsecsElapsed()/1000000000.0f;
        }
        loopTime.restart();

        {
            QMutexLocker locker(&pidMutex);
            // Simple PID-Controller
            // Source: http://rn-wissen.de/wiki/index.php?title=Regelungstechnik#PID-Regler

            // Compute error signal
            float e = 0;

            e = (energyXL*avgFlowVecXL-energyXR*avgFlowVecXR)/(energyXR+energyXL);

            // Compute integrated error
            eSum = qMax(-PUSHBOT_PID_MAX_ESUM,qMin(eSum + e,PUSHBOT_PID_MAX_ESUM));
            PRINT_DEBUG_FMT("dT: %f", deltaT);
            PRINT_DEBUG_FMT("e: %f",e);
            PRINT_DEBUG_FMT("ESum: %f",eSum);
            PRINT_DEBUG_FMT("P: %f",Kp*e);

            // Ignore differential part in first run
            if(deltaT > 0) {
                PRINT_DEBUG_FMT("I: %f",Ki*deltaT*eSum);
                PRINT_DEBUG_FMT("D: %f",Kd*(e-eOld)/deltaT);
                out = Kp*e + Ki*deltaT*eSum + Kd*(e-eOld)/deltaT;
            } else
                out = Kp*e;

            //if(!timer2.isValid())
            //   timer2.restart();

            //qDebug("%llu %f %f",timer2.nsecsElapsed()/1000,out, e);

            // Store last error
            eOld = e;
        }

    } else {
        // Don't generate output
        out = 0;
    }
    int speedLeft,speedRight;
    // Send commands to pushbot but clamp output if motor speed is out of range
    speedLeft = CLAMP((int)round(PUSHBOT_VELOCITY_DEFAULT - out/2),
                      PUSHBOT_VELOCITY_MIN,PUSHBOT_VELOCITY_MAX);
    speedRight = CLAMP((int)round(PUSHBOT_VELOCITY_DEFAULT + out/2),
                       PUSHBOT_VELOCITY_MIN,PUSHBOT_VELOCITY_MAX);

    // Execute commands or print to console
    if(robotInterface->isStreaming()) {
        emit setMotorVelocity(PUSHBOT_MOTOR_LEFT,speedLeft);
        emit setMotorVelocity(PUSHBOT_MOTOR_RIGHT,speedRight);
    } else {  // If not running, write debug info
        PRINT_DEBUG_FMT("[Control output] L: %d, r: %d",speedLeft,speedRight);
    }
}
