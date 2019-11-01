# Driving Pi
## Introduction
This app does autonomous LEGO car driving by classifying road segmentation and object detection for traffic light using Intel Movidius Neural Compute Stick 2.


## Prerequisites
The app requires the following hardware: 
1. 1 x Raspberry Pi 3B+*
2. 1 x Intel &reg; Neural Compute Stick 2 (NCS 2)
3. 1 x BrickPi 3 - DEXTER
4. 1 x Web camera (USB) OR CameraPi
5. LEGO MINDSTORMS EV3 KIT
6. BrickStuff (for traffic light)
7. LEGO Roads set

 ## Setup
 First, setup the Raspberry Pi3 and the BrickPi3: [Get started](https://www.dexterindustries.com/BrickPi/brickpi-tutorials-documentation/getting-started/)<br> After installing the boards, start building the LEGO car using LEGO MINDSTORMS EV3 KIT: [step-by-step](https://le-www-live-s.legocdn.com/sc/media/lessons/mindstorms-ev3/building-instructions/ev3-rem-driving-base-79bebfc16bd491186ea9c9069842155e.pdf) <br>
 Instead of using the LEGO Mindstorm Controller, attach the Pi+BrickPi to the car. <br>
 Build your LEGO road, and setup the traffic light on it.
 ___
 ![](src/docs/pic1.jpg)
 ![](src/docs/pic2.jpg)
 ![](src/docs/pic3.jpg)
 ___

## How to run the sample
To run the sample, change directory to the birds application folder and use the command: 
```
make run
```

Alternatively, run the Python script directly for more options:
```
python3 driving_pi.py -h
```
___
![](src/docs/pic4.gif)
___
## Makefile
Provided Makefile has various targets that help with the above mentioned tasks.

### make run
Runs the sample application.

### make help
Shows available targets.

### make all
Builds and/or gathers all the required files needed to run the application.

### make data
Gathers all of the required data need to run the sample.

### make deps
Builds all of the dependencies needed to run the sample.

### make install-reqs
Checks required packages that aren't installed as part of the OpenVINO installation. 

### make uninstall-reqs
Uninstalls requirements that were installed by the sample program.
 
### make clean
Removes all the temporary files that are created by the Makefile.