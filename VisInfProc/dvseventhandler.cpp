#include "dvseventhandler.h"
#include <QFile>

DVSEventHandler::DVSEventHandler(QObject *parent)
{
    connect(&timer,SIGNAL(timeout()),this,SLOT(onTimePlayback()));

    eventList.clear();
    eventIdx = 0;
}


bool DVSEventHandler::playBackFile(QString fileName, int speedMs)
{
    eventIdx = 0;
    QFile f(fileName);
    if(!f.open(QIODevice::ReadOnly)){
        return false;
    }
    QByteArray buff = f.readAll();

    if(buff.size() == 0)
        return false;

    // Parse events from file
    QString versionToken = "#!AER-DAT";
    int version = -1;

    // Parse header
    QList<QByteArray> lines = buff.split('\n');
    int lineIdx =  0;
    for(; lineIdx < lines.length(); lineIdx++){
        if(!lines.at(lineIdx).startsWith("#"))
            break;
        if(lines.at(lineIdx).contains(versionToken.toLocal8Bit())){
            QByteArray b = lines.at(lineIdx);
            b.remove(0,versionToken.length());
            b.chop(3);
            version = b.toInt();
        }
        qDebug(lines.at(lineIdx).toStdString().c_str());
    }
    qDebug(QString("Version: %1").arg(version).toLocal8Bit());

    int dataToSkip = 0;
    for(int i = 0; i < lineIdx; i++)
        dataToSkip += lines.at(i).length()+1;
    qDebug(QString("HeaderSize: %1").arg(dataToSkip).toLocal8Bit());
    // Remove header
    buff.remove(0,dataToSkip);
    // Extract events
    int numBytesPerEvent = 6;
    if(version == 2)
        numBytesPerEvent = 8;
    int eventCnt = buff.size()/numBytesPerEvent;
    qDebug(QString("Num Events: %1").arg(eventCnt).toLocal8Bit());
    int buffIdx = 0;
    for(int i = 0; i < eventCnt; i++){
        u_int32_t ad = 0,time = 0;
        switch(version){
        case 1:
            ad = ((uchar)buff.at(buffIdx++) << 0x10);
            ad |= ((uchar)buff.at(buffIdx++) << 0x00);

            time = ((uchar)buff.at(buffIdx++) << 0x40);
            time |= ((uchar)buff.at(buffIdx++) << 0x20);
            time |= ((uchar)buff.at(buffIdx++) << 0x10);
            time |= ((uchar)buff.at(buffIdx++) << 0x00);

            break;
        case 2:
            ad = ((uchar)buff.at(buffIdx++) << 0x40);
            ad |= ((uchar)buff.at(buffIdx++) << 0x20);
            ad |= ((uchar)buff.at(buffIdx++) << 0x10);
            ad |= ((uchar)buff.at(buffIdx++) << 0x00);

            time = ((uchar)buff.at(buffIdx++) << 0x40);
            time |= ((uchar)buff.at(buffIdx++) << 0x20);
            time |= ((uchar)buff.at(buffIdx++) << 0x10);
            time |= ((uchar)buff.at(buffIdx++) << 0x00);
            break;
        }
        // Extract event from address by assuming a DVS128 camera
        DVSEvent e;
        e.On = ad & 0x01;       // Polarity: LSB
        e.posX = ((ad >> 0x01) & 0xFE);
        e.posY = ((ad >> 0x10) & 0x7F) ;
        e.timestamp = time;
        eventList.append(e);
        qDebug(QString("%1 %2 %3 %4").arg(e.On).arg(e.posX).arg(e.posY).arg(e.timestamp).toLocal8Bit());
    }
    timer.start(speedMs);
    return true;
}

void DVSEventHandler::onTimePlayback()
{
    if(eventIdx < eventList.size()){
        DVSEvent e = eventList.at(eventIdx++);
        emit OnNewEvent(e);
    }

    if(eventIdx >= eventList.size())
    {
        eventIdx = 0;
        eventList.clear();
        timer.stop();
    }
}
