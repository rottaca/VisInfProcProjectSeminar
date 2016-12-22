#-------------------------------------------------
#
# Project created by QtCreator 2016-12-10T18:32:25
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = VisInfProc
TEMPLATE = app

QMAKE_CXXFLAGS += -fopenmp
LIBS += -fopenmp

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
    motionenergyestimator.cpp \
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
    motionenergyestimator.h \
    opticflowestimator.h

FORMS    += mainwindow.ui

# CUDA Settings
CUDA_SOURCES = cuda_stuff.cu

CUDA_DIR = /usr/local/cuda
CUDA_ARCH = sm_32 # as supported by the Tegra K1

INCLUDEPATH += $$CUDA_DIR/include
LIBS += -L $$CUDA_DIR/lib -lcudart -lcuda
osx: LIBS += -F/Library/Frameworks -framework CUDA

cuda.commands = $$CUDA_DIR/bin/nvcc -c -arch=$$CUDA_ARCH -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -M ${QMAKE_FILE_NAME}
cuda.input = CUDA_SOURCES
cuda.output = ${QMAKE_FILE_BASE}_cuda.o
QMAKE_EXTRA_COMPILERS += cuda

DISTFILES +=
