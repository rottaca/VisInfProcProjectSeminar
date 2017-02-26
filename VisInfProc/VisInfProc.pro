#-------------------------------------------------
#
# Project created by QtCreator 2016-12-10T18:32:25
#
#-------------------------------------------------

QT       += core gui network

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = VisInfProc
TEMPLATE = app

QMAKE_CXXFLAGS += -fopenmp -lnvToolsExt
LIBS += -fopenmp -lnvToolsExt
CONFIG += c++11

SOURCES += main.cpp\
        mainwindow.cpp \
    filtersettings.cpp \
    filtermanager.cpp \
    buffer1d.cpp \
    buffer2d.cpp \
    buffer3d.cpp \
    filterset.cpp \
    helper.cpp \
    motionenergyestimator.cpp \
    opticflowestimator.cpp \
    worker.cpp \
    basebuffer.cpp \
    pushbotcontroller.cpp \
    edvsinterface.cpp \
    eventbuilder.cpp \
    aspectratiopixmap.cpp \
    filterselectionform.cpp

HEADERS  += mainwindow.h \
    filtersettings.h \
    filtermanager.h \
    buffer1d.h \
    buffer2d.h \
    buffer3d.h \
    filterset.h \
    helper.h \
    motionenergyestimator.h \
    opticflowestimator.h \
    cuda_helper.h \
    cuda_settings.h \
    worker.h \
    basebuffer.h \
    settings.h \
    datatypes.h \
    pushbotcontroller.h \
    edvsinterface.h \
    eventbuilder.h \
    aspectratiopixmap.h \
    myqgraphicsimage.h \
    filterselectionform.h

FORMS    += mainwindow.ui \
    filterselectionform.ui

# CUDA Settings
CUDA_SOURCES =  cuda_helper.cu \
                cuda_filtermanager.cu \
                cuda_buffer.cu \
                cuda_motionenergy.cu \
                cuda_opticflow.cu

CUDA_DIR = /usr/local/cuda
CUDA_ARCH = sm_50

INCLUDEPATH += $$CUDA_DIR/include
LIBS += -L $$CUDA_DIR/lib64 -lcuda -lcudart
osx: LIBS += -F/Library/Frameworks -framework CUDA

cuda.commands = $$CUDA_DIR/bin/nvcc -c -arch=$$CUDA_ARCH -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME} -lineinfo
cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -M ${QMAKE_FILE_NAME}
cuda.input = CUDA_SOURCES
cuda.output = ${QMAKE_FILE_BASE}_cuda.o
QMAKE_EXTRA_COMPILERS += cuda

DISTFILES += \
    cuda_helper.cu \
    cuda_motionenergy.cu \
    cuda_opticflow.cu \
    cuda_filtermanager.cu \
    cuda_buffer.cu

RESOURCES += \
    res/resource.qrc
