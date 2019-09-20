# video_face_matcher
## Introduction
The video_face_matcher example app uses the TensorFlow [ncappzoo/networks/facenet](../../networks/facenet) neural network and the [ncappzoo/networks/face_detection_retail_0004](../../networks/face_detection_retail_0004) network to find a face in a video camera stream that matches with a known face image.

The provided video_face_matcher.py python app starts a webcam and shows the camera preview in a GUI window.  When the camera preview shows a face that matches the known valid face (video_face_matcher/validated_images/valid.jpg) a green frame is displayed around the person's face to indicate a match was found.  

![](screen_shot.png)

## Important Application details
There are two ways to make the app recognize any face you'd like. 

**Method one**: Run the app and use the mouse pointer to click on an unknown face. You will be prompted to enter the user's name. This will create an image on the disk in the validated_faces fodler and add the feature vector to the valid faces.

**Method two**: Add a sub-folder with images of the person you'd like to recognize to the validated_faces folder.  Make sure to name the sub-folder with the same name as the person you'd like to identify.  The application will use the folder's name when referring to the person. The names of the files inside of the named folder can be any name.  

**Note**: See the example inside of the validated_faces folder after running the command **make**. 

The application will work better with more validated images.

**Tip**: You can easily take pictures using a webcam using the Cheese application on Ubuntu.  When creating the images, please make sure that there is only one face per image.  If there are multiple faces in the validated image, the app will only take the first one it detects. 

The app will detect the face of the person in the images and create a 512 dimensional feature vector for each face.  It will then use these "validated" feature vectors to compare against faces that the app detects in the camera stream.  The app can be used with different people and multiple faces on the camera frame. 

To determine a match, the **FACE_MATCH_THRESHOLD** value is used.  You might want to adjust this value keeping in mind that the closer this value is to 0.0, the more exact and less forgiving the matching will be.  The initial value for **FACE_MATCH_THRESHOLD** is **1.10** but you will likely want to play with this value for your own needs.

The provided Makefile does the following:
1. Makes the facenet IR file from the ncappzoo/networks/facenet example and copies it to the base directory of this project.
2. Makes the face_detection_retail_0004 IR file from the ncappzoo/networks/face_detection_retail_0004 example and copies it to the base directory of this project.
3. Runs the provided video_face_matcher.py program that creates a GUI window which shows the camera stream and match/not match status.

## Prerequisites
This program requires:
- 1 NCS1/NCS2 device
- A webcam
- OpenVINO 2019 R2 or greater
- tkinter

## How to run
Open a terminal at the project root folder and run the command **make run**. The application uses the MYRIAD plugin as the default device. You can also run it on CPU by using the command **make run_cpu**.

Note: All development and testing has been done on Ubuntu 16.04 on an x86-64 machine.

## Makefile
Provided Makefile has various targets that help with the above mentioned tasks.

### make run or make run_py
Runs the sample application.

### make help
Shows available targets.

### make all
Builds and/or gathers all the required files needed to run the application.

### make data
Gathers all of the required data need to run the sample.

### make deps
Builds all of the dependencies needed to run the sample.

### make default_model
Compiles an IR file from a default model to be used when running the sample.

### make install-reqs
Checks required packages that aren't installed as part of the OpenVINO installation. The only required packages is tkinter.

### make uninstall-reqs
Uninstalls requirements that were installed by the sample program. 
 
### make clean
Removes all the temporary files that are created by the Makefile.

