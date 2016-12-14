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
    filter3d.cpp \
    filter1d.cpp \
    filter2d.cpp \
    filtermanager.cpp

HEADERS  += mainwindow.h \
    filtersettings.h \
    filter3d.h \
    filter1d.h \
    filter2d.h \
    filtermanager.h

FORMS    += mainwindow.ui
