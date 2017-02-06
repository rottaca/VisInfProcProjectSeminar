#ifndef DATATYPES_H
#define DATATYPES_H
#include <inttypes.h>

// Only stores the necessary information for further processing on gpu
typedef struct DVSEvent {
    uint8_t x;
    uint8_t y;
    uint32_t timestamp;
} DVSEvent;

#endif // DATATYPES_H
