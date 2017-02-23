#ifndef EVENTBUILDER_H
#define EVENTBUILDER_H

#include <QtGlobal>

#include "datatypes.h"

class EventBuilder
{
public:

    typedef enum AddressVersion {Addr2Byte = 2,Addr4Byte = 4} AddressVersion;
    typedef enum TimestampVersion {Time4Byte = 4, Time3Byte = 3,
                                   Time2Byte = 2, TimeDelta = -1,
                                   TimeNoTime = 0
                                  } TimestampVersion;

    EventBuilder();
    ~EventBuilder();
    /**
     * @brief initEvBuilder Initializes the event builder for a specific address and timestamp format
     * @param addrVers
     * @param timeVers
     */
    void initEvBuilder(AddressVersion addrVers, TimestampVersion timeVers);
    /**
     * @brief evBuilderProcessNextByte Processes the next character from the file or online-stream.
     * Returns true, when a new event is constructed. Its data stored in the passed object reference.
     * @param c
     * @param event
     * @return
     */
    bool evBuilderProcessNextByte(char c, DVSEvent &event);
    DVSEvent evBuilderParseEvent();

private:
    // Data for the event builder
    TimestampVersion evBuilderTimestampVersion;
    AddressVersion evBuilderAddressVersion;
    // Current byte index for the next event
    int evBuilderByteIdx;
    // The buffer size for the event builder
    int evBuilderBufferSz;
    // Pointer to the event builder data (currently stored bytes for the next event)
    char* evBuilderData;
    // Timestamp for events with delta times
    quint32 evBuilderSyncTimestamp;
    // Timestamp of last event to detect overflows
    quint32 evBuilderLastTimestamp;
    //QMutex evBuilderMutex;
};

#endif // EVENTBUILDER_H
