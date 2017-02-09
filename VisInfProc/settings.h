#ifndef SETTINGS_H
#define SETTINGS_H

/*****************************************************************
// Camera Section
*****************************************************************/
// DVS camera resolution
#define DVS_RESOLUTION_WIDTH 128
#define DVS_RESOLUTION_HEIGHT 128

/*****************************************************************
// GUI
*****************************************************************/
// GUI refresh rate
#define GUI_RENDERING_FPS 25
// Refreshrate of statistics
#define GUI_STAT_UPDATE_FPS 5

/*****************************************************************
// CUDA
*****************************************************************/
// Threads in a single block
#define THREADS_PER_BLOCK 64
// Maximum number of events in shared memory
#define MAX_SHARED_GPU_EVENTS 128

/*****************************************************************
// OpticFlow estimator
*****************************************************************/
// Spatial resolution of garbor filter (odd value)
#define FILTER_SPATIAL_SIZE_PX 11
// Temporal resolution of time function between 0 and TEMPORAL_END
#define FILTER_TEMPORAL_RES 20
// End value x of temporal function t(x)
#define FILTER_TEMPORAL_END 0.7f
// Additional filter settings located in filtersettings.cpp
#define FLOW_DEFAULT_MIN_ENERGY_THRESHOLD 0.25f

/*****************************************************************
// Pushbot
*****************************************************************/
// Pushbot commands
#define CMD_SET_TIMESTAMP_MODE "!E1\n"      // Do not change streaming mode !
#define CMD_ENABLE_EVENT_STREAMING "!E+\n"
#define CMD_DISABLE_EVENT_STREAMING "!E-\n"
#define CMD_ENABLE_MOTORS "!M+\n"
#define CMD_DISABLE_MOTORS "!M-\n"
#define CMD_SET_VELOCITY "!MV%1=%2\n"
#define CMD_RESET_BOARD "R\n"
// Refresh rate of optic flow processing
#define PUSH_BOT_PROCESS_FPS 15
// Motor Velocity minimum for pid control (0-100)
#define PUSHBOT_VELOCITY_MIN 5
// Motor Velocity maximum for pid control
#define PUSHBOT_VELOCITY_MAX 50
// Pushbot motor number left
#define PUSHBOT_MOTOR_LEFT 0
// Pushbot motor number right
#define PUSHBOT_MOTOR_RIGHT 1
// Default speed
#define PUSHBOT_VELOCITY_DEFAULT 30
// Default PID values for pushbot PID controller
#define PUSHBOT_PID_P_DEFAULT 0.1
#define PUSHBOT_PID_I_DEFAULT 0.0
#define PUSHBOT_PID_D_DEFAULT 0.0
// Maximum absolute integrated error
#define PUSHBOT_PID_MAX_ESUM 100.f

// Minimum detection energy
// Summed energy over an image half hast to be greater as the value below
// Otherwise not steering signal is generated
#define PUSHBOT_MIN_DETECTION_ENERGY 80

/*****************************************************************
// Debug Section
*****************************************************************/
// Uses the direction channel to encode the interpolation state
// (Interpolate, Out of range, no interpolation,...)
// Replaces optic flow and disables push bot control
//#define DEBUG_FLOW_DIR_ENCODE_INTERPOLATION

#ifndef DEBUG_FLOW_DIR_ENCODE_INTERPOLATION
// Disables the interpolation between speeds and takes the maximum response
#define DISABLE_INTERPOLATION
#endif

/*****************************************************************
// Other
*****************************************************************/
// Convolution buffers per filter orientation
// DO NOT CHANGE
#define FILTERS_PER_ORIENTATION 2
// Maximum wait time for stopping a thread
#define THREAD_WAIT_TIME_MS 1000

/*****************************************************************
// Macros
*****************************************************************/
#define CLAMP(v,mn,mx) qMin(mx,qMax(mn,v))

#ifdef QT_DEBUG
#define PRINT_DEBUG(msg) qDebug(msg)
#define PRINT_DEBUG_FMT(format, ...) qDebug(format,__VA_ARGS__)
#else
#define PRINT_DEBUG(msg) {}
#define PRINT_DEBUG_FMT(format, ...) {}
#endif
#endif // SETTINGS_H
