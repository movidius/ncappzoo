# gender_age_lbp: A Movidius Neural Compute Stick example for gender and age inference

This app does facial detection and age/gender inference using the Intel Neural Compute Stick. This demo was tested with the Movidius NCSDK Build 1.12.

# Prerequisites

This app requires the following components are available:
1. Movidius Neural Compute Stick
2. Movidius the Neural Compute SDK
3. A webcam (laptop or usb)
4. OpenCV 

* Note: You can install OpenCV by using the install-opencv-from_source.sh script which is included with the ncappzoo app "street_cam_threaded"

# Building the example

To run the example code do the following :
1. Open a terminal and change directory to the gender_age_lbp example base directory
2. Type the following command in the terminal: make all

# Running the Example

After building the example you can run the example code by doing the following :
1. Open a terminal and change directory to the gender_age_lbp base directory
2. Type the following command in the terminal: make run 

When the application runs normally, another window should pop up and show the feed from the webcam. The program should perform inferences on faces on frames taken from the webcam/usb cam.

* If you only have one stick, you can use "make run_gender" or "make run_age" to run a single network instead.

# Keybindings 

Controls while program is running. 

* Escape : Quit the program.

# Additional Information

At the moment, the program is limited to detecting and making an inference on one person at a time. 
