#!/bin/bash

VIDEOIO=`pkg-config --libs opencv | grep "lopencv_videoio"`
if [ "$VIDEOIO" == "" ]
then
    echo "no videoio library"
    g++ -std=c++11 cpp/gender_age_lbp.cpp cpp/fp16.c -o cpp/gender_age_lbp -L/usr/local/lib -lmvnc -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_objdetect
else
    echo "found videoio library"
    g++ -std=c++11 cpp/gender_age_lbp.cpp cpp/fp16.c -o cpp/gender_age_lbp -L/usr/local/lib -lmvnc -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_objdetect -lopencv_videoio
fi
