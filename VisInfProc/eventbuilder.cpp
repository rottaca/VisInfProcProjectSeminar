#include "eventbuilder.h"

EventBuilder::EventBuilder()
{
    evBuilderData = NULL;
    evBuilderAddressVersion = Addr2Byte;
    evBuilderTimestampVersion = TimeNoTime;
    evBuilderBufferSz = 0;
    evBuilderByteIdx = 0;
    evBuilderSyncTimestamp = 0;
}

EventBuilder::~EventBuilder()
{
    if(evBuilderData != NULL)
        delete[] evBuilderData;
    evBuilderData = NULL;
}

void EventBuilder::initEvBuilder(AddressVersion addrVers, TimestampVersion timeVers)
{
    //QMutexLocker locker(&evBuilderMutex);
    evBuilderTimestampVersion = timeVers;
    evBuilderAddressVersion = addrVers;
    evBuilderByteIdx = 0;
    evBuilderSyncTimestamp = 0;
    evBuilderLastTimestamp = 0;

    switch(addrVers) {
    case Addr2Byte:
        evBuilderBufferSz = 2;
        break;
    case Addr4Byte:
        evBuilderBufferSz = 4;
    }

    switch(timeVers) {
    case Time4Byte:
        evBuilderBufferSz += 4;
        break;
    case Time3Byte:
        evBuilderBufferSz += 3;
        break;
    case Time2Byte:
        evBuilderBufferSz += 2;
        break;
    case TimeNoTime:
        evBuilderBufferSz += 0;
        break;
    case TimeDelta:
        evBuilderBufferSz += 4;
        break;
    }

    if(evBuilderData != NULL)
        delete[] evBuilderData;
    evBuilderData = new char[evBuilderBufferSz];
}

bool EventBuilder::evBuilderProcessNextByte(char c, DVSEvent &event)
{
    //QMutexLocker locker(&evBuilderMutex);
    // Simple sync check: Check if MSB of first byte is 1, otherwise skip
    //if(evBuilderByteIdx == 0 && !(c & 0x80)) {
    //    qWarning("Skipped event byte!");
    //    return false;
    //}

    // Store byte in buffer
    evBuilderData[evBuilderByteIdx++] = c;
    if(evBuilderTimestampVersion == TimeDelta) {
        // addressbytes done ?
        if(evBuilderByteIdx >= evBuilderAddressVersion) {
            // Check for leading 1 in timestamp bytes
            if(c & 0x80) {
                event = evBuilderParseEvent();
                evBuilderByteIdx = 0;
                return true;
            } else if(evBuilderByteIdx == evBuilderBufferSz) {
                qCritical("Event not recognized! Skipped %d data bytes! "
                          "Please restart!",evBuilderBufferSz);
                evBuilderByteIdx = 0;
            }
        }
    } else {
        // Buffer full ? Event ready
        if(evBuilderByteIdx == evBuilderBufferSz) {
            event = evBuilderParseEvent();
            evBuilderByteIdx = 0;
            return true;
        }
    }
    return false;
}

DVSEvent EventBuilder::evBuilderParseEvent()
{
    // TODO: Check endianess from system
    // Event data is MBS first

    u_int32_t ad = 0,time = 0;
    int idx = 0;
    int addrBytes = evBuilderAddressVersion;
    switch (evBuilderAddressVersion) {
    case Addr2Byte:
        ad |= uint32_t((uchar)evBuilderData[idx++] << 0x08);
        ad |= uint32_t((uchar)evBuilderData[idx++] << 0x00);
        break;
    case Addr4Byte:
        ad = uint32_t((uchar)evBuilderData[idx++] << 0x18);
        ad |= uint32_t((uchar)evBuilderData[idx++] << 0x10);
        ad |= uint32_t((uchar)evBuilderData[idx++] << 0x08);
        ad |= uint32_t((uchar)evBuilderData[idx++] << 0x00);
        break;
    }
    // TODO Use evBuilderSyncTimestamp for all types of timestamps to avoid overflows in time
    switch(evBuilderTimestampVersion) {
    case Time4Byte:
        time = uint32_t((uchar)evBuilderData[idx++] << 0x18);
        time |= uint32_t((uchar)evBuilderData[idx++] << 0x10);
        time |= uint32_t((uchar)evBuilderData[idx++] << 0x08);
        time |= uint32_t((uchar)evBuilderData[idx++] << 0x00);
        // No overflow handling when using 4 bytes
        break;
    case Time3Byte:
        time = uint32_t((uchar)evBuilderData[idx++] << 0x10);
        time |= uint32_t((uchar)evBuilderData[idx++] << 0x08);
        time |= uint32_t((uchar)evBuilderData[idx++] << 0x00);
        // Timestamp overflow ?
        if(time < evBuilderLastTimestamp) {
            evBuilderSyncTimestamp += 0xFFFFFF;
        }
        time += evBuilderSyncTimestamp;
        break;
    case Time2Byte:
        time = uint32_t((uchar)evBuilderData[idx++] << 0x08);
        time |= uint32_t((uchar)evBuilderData[idx++] << 0x00);
        // Timestamp overflow ?
        if(time < evBuilderLastTimestamp) {
            evBuilderSyncTimestamp += 0xFFFF;
        }
        time += evBuilderSyncTimestamp;
        break;
    case TimeDelta: {
        // TODO Check
        // Parse variable timestamp
        // Store bytes in flipped order in time variable
        int pos = (evBuilderByteIdx-1)*7;
        for(int j = 0; j < evBuilderByteIdx-addrBytes; j++) {
            time |= uint32_t(((uchar)evBuilderData[idx++] & 0x7F) << pos);
            pos-=7;
        }
        // Convert relative to absolute timestamp
        evBuilderSyncTimestamp += time;
        time = evBuilderSyncTimestamp;
        break;
    }
    case TimeNoTime:
        time = 0;
        break;
    }

    evBuilderLastTimestamp = time;

    DVSEvent e;
    // Extract event from address by assuming a DVS128 camera
    //e.On = ad & 0x01;       // Polarity: LSB
    // flip axis to match qt's image coordinate system
    e.x = 127 - ((ad >> 0x01) & 0x7F);  // X: 0 - 127
    e.y = 127 - ((ad >> 0x08) & 0x7F) ; // Y: 0 - 127
    e.timestamp = time;

    return e;
}
