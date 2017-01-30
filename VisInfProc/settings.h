#ifndef SETTINGS_H
#define SETTINGS_H

/*****************************************************************
// Camera Section
*****************************************************************/
// DVS camera resolution
#define DVS_RESOLUTION_WIDTH 128
#define DVS_RESOLUTION_HEIGHT 128

/*****************************************************************
// General
*****************************************************************/
// GUI refresh rate
#define FPS 25
// Refresh rate of optic flow processing
#define PUSH_BOT_PROCESS_FPS 15

/*****************************************************************
// Pushbot
*****************************************************************/
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
// Commands
#define CMD_SET_TIMESTAMP_MODE "!E1\n"      // Do not change streaming mode !
#define CMD_ENABLE_EVENT_STREAMING "!E+\n"
#define CMD_DISABLE_EVENT_STREAMING "!E-\n"
#define CMD_ENABLE_MOTORS "!M+\n"
#define CMD_DISABLE_MOTORS "!M-\n"
#define CMD_SET_VELOCITY "!MV%1=%2\n"
#define CMD_RESET_BOARD "R\n"

// Minimum detection energy
// Summed energy over an image half hast to be greater as the value below
// Otherwise not steering signal is generated
#define PUSHBOT_MIN_DETECTION_ENERGY 80

/*****************************************************************
// Debug Section
*****************************************************************/
// Uses the direction channel to encode the interpolation state
// (Interpolate, Out of range, no interpolation,...)
// Destroys optic flow and disables push bot control
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

/*****************************************************************
// Macros
*****************************************************************/
#define CLAMP(v,mn,mx) qMin(mx,qMax(mn,v))

#endif // SETTINGS_H
