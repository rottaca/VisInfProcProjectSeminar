#ifndef SETTINGS_H
#define SETTINGS_H

// DVS camera resolution
#define DVS_RESOLUTION_WIDTH 128
#define DVS_RESOLUTION_HEIGHT 128

// GUI refresh rate
#define FPS 25
// Refresh rate of optic flow processing
#define PUSH_BOT_PROCESS_FPS 10

// Convolution buffers per filter orientation
// DO NOT CHANGE
#define FILTERS_PER_ORIENTATION 2


//#include "cuda_settings.h"

/*****************************************************************
// Debug Section
*****************************************************************/
// Uses the direction channel to encode the interpolation state
// (Interpolate, Out of range, no interpolation,...)
// Destroys optic flow and disables push bot control
//#define DEBUG_FLOW_DIR_ENCODE_INTERPOLATION

// Disables the interpolation between speeds and takes the maximum response
#define DISABLE_INTERPOLATION


#endif // SETTINGS_H

