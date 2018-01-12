# gender_age_lbp 

This app does facial detection and age/gender inference using the Intel NCS. This demo was tested with the Movidius NC SDK Build 1.11.

# Prerequisites

This code sample requires installing OpenCV 3.1+. Follow these directions to install the necessary prerequisites.


### OpenCV

First, download required libraries with the following command:
* `sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libx11-dev libgtk2.0-dev`

##### Get OpenCV

* `cd ~`
* `git clone https://github.com/opencv/opencv.git`
* `cd opencv`
* `git checkout 3.1.0`

##### Build OpenCV

* `mkdir release`
* `cd release`
* `cmake -D CMAKE_BUILD_TYPE=RELEASE -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D WITH_GTK=ON -D WITH_GTK_2_X=ON -D CMAKE_INSTALL_PREFIX=/usr ../`
* `make -j4`
* `sudo make install`

#### Copy Additional Files

This app also requires the OpenCV file lbpcascade_frontalface_improved.xml available at the OpenCV github page. You can download it by typing in the following command:

* `cd ~`
* `wget https://github.com/opencv/opencv/blob/master/data/lbpcascades/lbpcascade_frontalface_improved.xml`
* `sudo mv lbp_frontalface_improved.xml /usr/share/OpenCV/lbpcascades/`

# How to build the sample

To build the sample, please type the following commands in the sample's root directory

1. mkdir build
2. cd build
3. cmake ..
4. make 

Alternatively, the build.sh script can be used to build the sample.

# Steps to execute

./gender_age_lbp


# Keybindings 

Controls while program is running. 

* Escape : Quit the program.

# Additional Information

Make sure that the gender and age stat.txt, graph file and category files are up to date in main.cpp.

#License

