#include "eventbuilder.h"

#include "settings.h"
EventBuilder::EventBuilder()
{
    evBuffer = NULL;
    addressVersion = Addr2Byte;
    timestampVersion = TimeNoTime;
    bufferSz = 0;
    byteIdx = 0;
    syncTimestamp = 0;
}

EventBuilder::~EventBuilder()
{
    if(evBuffer != NULL)
        delete[] evBuffer;
    evBuffer = NULL;
}

void EventBuilder::initEvBuilder(AddressVersion addrVers, TimestampVersion timeVers)
{
    //QMutexLocker locker(&evBuilderMutex);
    timestampVersion = timeVers;
    addressVersion = addrVers;
    byteIdx = 0;
    syncTimestamp = 0;
    lastTimestamp = 0;

    switch(addrVers) {
    case Addr2Byte:
        bufferSz = 2;
        break;
    case Addr4Byte:
        bufferSz = 4;
    }

    switch(timeVers) {
    case Time4Byte:
        bufferSz += 4;
        break;
    case Time3Byte:
        bufferSz += 3;
        break;
    case Time2Byte:
        bufferSz += 2;
        break;
    case TimeNoTime:
        bufferSz += 0;
        break;
    case TimeDelta:
        bufferSz += 4;
        break;
    }

    if(evBuffer != NULL)
        delete[] evBuffer;
    evBuffer = new char[bufferSz];
}

bool EventBuilder::evBuilderProcessNextByte(char c, DVSEvent &event, bool onlineMode)
{
    //QMutexLocker locker(&evBuilderMutex);
    if(onlineMode) {
        // Simple sync check: Check if MSB of first byte is 1, otherwise skip
        // Address Mode is always 2 Byte, in Online Mode
        if(byteIdx == 0 && !(c & 0x80)) {
            qWarning("[EventBuilder] Invalid first byte! Skipped one byte!");
            return false;
        }
    }
    // Store byte in buffer
    evBuffer[byteIdx++] = c;
    if(timestampVersion == TimeDelta) {
        // address bytes received ?
        if(byteIdx >= addressVersion) {
            // Check for leading 1 in timestamp bytes
            if(c & 0x80) {
                return evBuilderParseEvent(onlineMode,event);
            } else if(byteIdx == bufferSz) {
                qCritical("[EventBuilder] Delta timestamp not detected! Skipped %d data bytes!",
                          bufferSz);
                byteIdx = 0;
            }
        }
    } else {
        // Buffer full ? Event ready
        if(byteIdx == bufferSz)
            return evBuilderParseEvent(onlineMode,event);
    }
    return false;
}

bool EventBuilder::evBuilderParseEvent(bool onlineMode, DVSEvent &e)
{
    // Event data is big-endian (highest byte first)
    // Convert to little-endian
    u_int32_t ad = 0,time = 0;
    int idx = 0;
    int addrBytes = addressVersion;
    switch (addressVersion) {
    case Addr2Byte:
        ad |= uint32_t((uchar)evBuffer[idx++] << 0x08);
        ad |= uint32_t((uchar)evBuffer[idx++] << 0x00);
        break;
    case Addr4Byte:
        ad = uint32_t((uchar)evBuffer[idx++] << 0x18);
        ad |= uint32_t((uchar)evBuffer[idx++] << 0x10);
        ad |= uint32_t((uchar)evBuffer[idx++] << 0x08);
        ad |= uint32_t((uchar)evBuffer[idx++] << 0x00);
        break;
    }
    switch(timestampVersion) {
    case Time4Byte:
        time = uint32_t((uchar)evBuffer[idx++] << 0x18);
        time |= uint32_t((uchar)evBuffer[idx++] << 0x10);
        time |= uint32_t((uchar)evBuffer[idx++] << 0x08);
        time |= uint32_t((uchar)evBuffer[idx++] << 0x00);
        // No overflow handling when using 4 bytes
        // Requires 64 bit integers, increases data
        break;
    case Time3Byte:
        time = uint32_t((uchar)evBuffer[idx++] << 0x10);
        time |= uint32_t((uchar)evBuffer[idx++] << 0x08);
        time |= uint32_t((uchar)evBuffer[idx++] << 0x00);
        // Timestamp overflow ?
        if(time < lastTimestamp) {
            syncTimestamp += 0xFFFFFF;
            PRINT_DEBUG("Timestmap warped!");
            PRINT_DEBUG_FMT("Sync: %u",syncTimestamp);
        }
        time += syncTimestamp;
        break;
    case Time2Byte:
        time = uint32_t((uchar)evBuffer[idx++] << 0x08);
        time |= uint32_t((uchar)evBuffer[idx++] << 0x00);
        // Timestamp overflow ?
        if(time < lastTimestamp) {
            syncTimestamp += 0xFFFF;
            PRINT_DEBUG("Timestmap warped!");
            PRINT_DEBUG_FMT("Sync: %u",syncTimestamp);
        }
        time += syncTimestamp;
        break;
    case TimeDelta: {
        // Parse variable timestamp: Store bytes in flipped order in time variable,
        // concat only the lowest 7 bits per byte
        int timestampBytes = byteIdx-addrBytes;
        // Most significant byte first
        int pos = (timestampBytes-1)*7;
        for(int j = 0; j < timestampBytes; j++) {
            time |= uint32_t(((uchar)evBuffer[idx++] & 0x7F) << pos);
            pos-=7;
        }
        // Convert relative to absolute timestamp
        time += syncTimestamp;
        // Store new sync timestamp
        syncTimestamp = time;
        break;
    }
    // Format contains no timestamp
    case TimeNoTime:
        time = 0;
        break;
    }

    lastTimestamp = time;

    // Extract event from address by assuming a DVS128 camera
    // Wierd event format !?
    if(!onlineMode) {
        // Format:
        //   Bit 0: On/Off
        //   Bit 7-1: X
        //   Bit 8-14: Y
        //   Bit 15: External Event, Unused
        //e.On = ad & 0x01;       // Polarity: LSB
        e.x = ((ad >> 0x01) & 0x007F);  // X: 0 - 127
        e.y = ((ad >> 0x08) & 0x007F); // Y: 0 - 127
        e.timestamp = time;
    } else {
        // Format:
        // Bit 0-6: Y
        // Bit 7: Unused
        // Bit 8-14: X
        // Bit 15: On/Off
        //e.On = (ad >> 0x08) & 0x01;       // Polarity: LSB
        e.x = ((ad >> 0x00) & 0x007F); // Y: 0 - 127
        e.y = 127 - ((ad >> 0x08) & 0x007F);  // X: 0 - 127
        e.timestamp = time;
    }
    // Reset buffer
    byteIdx = 0;
    return true;
}
