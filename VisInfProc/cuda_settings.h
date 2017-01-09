#ifndef CUDA_SETTINGS_H
#define CUDA_SETTINGS_H

#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 1024


#define GPU_LUT_COLORMAP_SZ 256
__constant__ unsigned char GPUrgbColormapLUT[GPU_LUT_COLORMAP_SZ*3] = {
    0,0,255,
    0,2,253,
    0,4,251,
    0,6,249,
    0,8,247,
    0,10,245,
    0,12,243,
    0,14,241,
    0,16,239,
    0,18,237,
    0,20,235,
    0,22,233,
    0,24,231,
    0,26,229,
    0,28,227,
    0,30,225,
    0,32,223,
    0,34,221,
    0,36,219,
    0,38,217,
    0,40,215,
    0,43,213,
    0,45,211,
    0,47,209,
    0,49,207,
    0,51,205,
    0,53,203,
    0,55,201,
    0,57,199,
    0,59,197,
    0,61,195,
    0,63,193,
    0,65,191,
    0,67,189,
    0,69,187,
    0,71,185,
    0,73,183,
    0,75,181,
    0,77,179,
    0,79,177,
    0,81,175,
    0,83,173,
    0,85,171,
    0,87,169,
    0,89,167,
    0,91,165,
    0,93,163,
    0,95,161,
    0,97,159,
    0,99,157,
    0,101,155,
    0,103,153,
    0,105,151,
    0,107,149,
    0,109,147,
    0,111,145,
    0,113,143,
    0,115,141,
    0,117,139,
    0,119,137,
    0,121,135,
    0,123,133,
    0,126,131,
    0,128,129,
    0,130,127,
    0,132,125,
    0,134,123,
    0,136,121,
    0,137,119,
    0,139,117,
    0,141,115,
    0,143,113,
    0,145,111,
    0,147,109,
    0,149,107,
    0,151,105,
    0,153,103,
    0,155,101,
    0,157,99,
    0,159,97,
    0,161,95,
    0,163,93,
    0,165,91,
    0,167,89,
    0,169,87,
    0,171,85,
    0,173,83,
    0,175,81,
    0,177,79,
    0,179,77,
    0,181,75,
    0,183,73,
    0,185,71,
    0,187,69,
    0,189,67,
    0,191,65,
    0,193,63,
    0,195,61,
    0,197,59,
    0,199,57,
    0,201,55,
    0,203,53,
    0,205,51,
    0,207,49,
    0,209,47,
    0,211,45,
    0,213,43,
    0,215,41,
    0,217,39,
    0,219,37,
    0,221,35,
    0,223,33,
    0,225,31,
    0,227,29,
    0,229,27,
    0,231,25,
    0,233,23,
    0,235,21,
    0,237,19,
    0,239,17,
    0,241,15,
    0,243,13,
    0,245,11,
    0,247,9,
    0,249,7,
    0,251,5,
    0,253,3,
    0,255,1,
    1,255,0,
    3,253,0,
    5,251,0,
    7,249,0,
    9,247,0,
    11,245,0,
    13,242,0,
    15,240,0,
    17,238,0,
    19,236,0,
    21,234,0,
    23,232,0,
    25,230,0,
    27,228,0,
    29,226,0,
    31,224,0,
    33,222,0,
    35,220,0,
    37,218,0,
    39,216,0,
    41,214,0,
    43,212,0,
    45,210,0,
    47,208,0,
    49,206,0,
    51,204,0,
    53,201,0,
    55,199,0,
    57,197,0,
    59,195,0,
    61,193,0,
    63,191,0,
    65,189,0,
    67,187,0,
    69,185,0,
    71,183,0,
    73,181,0,
    75,179,0,
    77,177,0,
    79,175,0,
    81,173,0,
    82,171,0,
    84,169,0,
    86,167,0,
    88,165,0,
    90,163,0,
    92,161,0,
    94,158,0,
    96,156,0,
    98,154,0,
    100,152,0,
    102,150,0,
    104,148,0,
    106,146,0,
    108,144,0,
    110,142,0,
    112,140,0,
    114,138,0,
    116,136,0,
    118,134,0,
    120,132,0,
    122,130,0,
    124,128,0,
    126,126,0,
    128,124,0,
    130,122,0,
    132,120,0,
    134,118,0,
    136,116,0,
    138,114,0,
    140,112,0,
    142,110,0,
    144,108,0,
    146,106,0,
    148,104,0,
    151,102,0,
    153,100,0,
    155,98,0,
    157,96,0,
    159,94,0,
    161,92,0,
    163,90,0,
    165,88,0,
    167,86,0,
    169,84,0,
    171,82,0,
    173,81,0,
    175,79,0,
    177,77,0,
    179,75,0,
    181,73,0,
    183,71,0,
    185,69,0,
    187,67,0,
    189,65,0,
    191,63,0,
    193,61,0,
    195,59,0,
    197,57,0,
    199,55,0,
    201,53,0,
    203,51,0,
    205,49,0,
    207,47,0,
    209,45,0,
    211,43,0,
    213,41,0,
    215,39,0,
    217,37,0,
    219,35,0,
    221,33,0,
    223,31,0,
    225,29,0,
    227,27,0,
    229,26,0,
    231,24,0,
    234,22,0,
    236,20,0,
    238,18,0,
    240,16,0,
    242,14,0,
    244,12,0,
    246,10,0,
    248,8,0,
    250,6,0,
    252,4,0,
    254,2,0,
    255,0,0
};

#endif // CUDA_SETTINGS_H

