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

    Kp = PUSHBOT_PID_P_DEFAULT;
    Ki = PUSHBOT_PID_I_DEFAULT;
    Kd = PUSHBOT_PID_D_DEFAULT;
    out = 0;
    eOld = 0;
    eSum = 0;

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
    if(processIntervalTimer.isActive())
        emit stopTimer();

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
    emit enableMotors(true);
    emit setMotorVelocity(0,PUSHBOT_VELOCITY_DEFAULT);
    emit setMotorVelocity(1,PUSHBOT_VELOCITY_DEFAULT);
    out = 0;
    eOld = 0;
    eSum = 0;
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

#ifdef DEBUG_FLOW_DIR_ENCODE_INTERPOLATION
    return;
#endif
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
    float cntL = 0,cntR = 0;

    #pragma omp parallel for
    for(int j = 0; j < sx*sy; j++) {
        float dir = flowDirPtr[j];
        float s = flowSpeedPtr[j];
        float e = flowEnergyPtr[j];
        if(s > 0) {
            // Left or right image half
            if(j % sx < sx/2) {
                avgFlowVecXL += cos(dir)*s*e;
                avgFlowVecYL += sin(dir)*s*e;
                cntL+=e;
            } else {
                avgFlowVecXR += cos(dir)*s*e;
                avgFlowVecYR += sin(dir)*s*e;
                cntR+=e;
            }
        }
    }

    // Normalize
    bool leftFlowValid = cntL > PUSHBOT_MIN_DETECTION_ENERGY;
    bool rightFlowValid = cntR > PUSHBOT_MIN_DETECTION_ENERGY;

    if(leftFlowValid) {
        avgFlowVecXL /= cntL;
        avgFlowVecYL /= cntL;
    } else {
        avgFlowVecXL = 0;
        avgFlowVecYL = 0;
    }
    if(rightFlowValid) {
        avgFlowVecXR /= cntR;
        avgFlowVecYR /= cntR;
    } else {
        avgFlowVecXR = 0;
        avgFlowVecYR = 0;
    }

    // Steering signal is useless if only left or
    // right motion is used for computation
    if(leftFlowValid && rightFlowValid) {

        // Get elapsed time
        float deltaT;
        if(!loopTime.isValid())
            deltaT = 0;
        else {
            deltaT = loopTime.elapsed()/1000.0f;
        }
        loopTime.restart();

        int speedLeft,speedRight;
        {
            QMutexLocker locker(&pidMutex);
            // Simple PID-Controller
            // Source: http://rn-wissen.de/wiki/index.php?title=Regelungstechnik#PID-Regler
            // Difference between horizontal flow on left and right half is error signal
            float e = avgFlowVecXL-avgFlowVecXR;
            // Compute integrated error
            eSum = qMax(-PUSHBOT_PID_MAX_ESUM,qMin(eSum + e,PUSHBOT_PID_MAX_ESUM));
            // Ignore differential part in first run
            if(deltaT > 0)
                out = Kp*e + Ki*deltaT*eSum + Kd*(e-eOld)/deltaT;
            else
                out = Kp*e;
            // Store last error
            eOld = e;

            // Send commands to pushbot but clamp output if motor speed is out of range
            speedLeft = CLAMP((int)round(PUSHBOT_VELOCITY_DEFAULT + out/2),
                              PUSHBOT_VELOCITY_MIN,PUSHBOT_VELOCITY_MAX);
            speedRight = CLAMP((int)round(PUSHBOT_VELOCITY_DEFAULT - out/2),
                               PUSHBOT_VELOCITY_MIN,PUSHBOT_VELOCITY_MAX);
        }

        if(robotInterface->isStreaming()) {
            emit setMotorVelocity(PUSHBOT_MOTOR_LEFT,speedLeft);
            emit setMotorVelocity(PUSHBOT_MOTOR_RIGHT,speedRight);
        } else {  // If not running, write debug info
            PRINT_DEBUG_FMT("[Control output] L: %d, r: %d",speedLeft,speedRight);
        }
    } else {
        // Don't generate output
        out = 0;
    }
}
