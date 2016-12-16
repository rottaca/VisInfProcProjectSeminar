#-------------------------------------------------
#
# Project created by QtCreator 2016-12-10T18:32:25
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = VisInfProc
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    filtersettings.cpp \
    filtermanager.cpp \
    buffer1d.cpp \
    buffer2d.cpp \
    buffer3d.cpp \
    filterset.cpp \
    helper.cpp \
    convolution3d.cpp \
    dvseventhandler.cpp \
    opticflowestimator.cpp

HEADERS  += mainwindow.h \
    filtersettings.h \
    filtermanager.h \
    buffer1d.h \
    buffer2d.h \
    buffer3d.h \
    filterset.h \
    helper.h \
    convolution3d.h \
    dvseventhandler.h \
    opticflowestimator.h

FORMS    += mainwindow.ui
