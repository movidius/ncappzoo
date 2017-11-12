# Introduction
The stream_ty_gn application is a object recognition and classification program.  It uses two the NCS devices along with TinyYolo and GoogLeNet to take a video stream and for each frame of video first it will get bounding boxes for objects via TinyYolo then it will use GoogLeNet to further classify the objects found.  It depends on the caffe/TinyYolo and caffe/GoogLeNet appzoo examples.
The provided Makefile does the following:
1. Builds both TinyYolo and GoogLeNet from their respective directories within the repo.
2. Copies the built NCS graph files from TinyYolo and GoogLeNet to the current directory.
3. Runs the provided stream_ty_gn.py program that creates a GUI window that shows the camera stream along with boxes around the identified objects. 

# How it Works
Two NCS devices are needed to run this application, one executes inferences for the Tiny Yolo network and one executes inferences for the googlenet network.  OpenCV is used to open a USB camera which is assumed to be attached to the host computer.  For each frame of video that the camera returns the program first runs a Tiny Yolo inference to find all objects in the image.  This will be limited to the 20 categories that Tiny Yolo recognizes.  Those categories can be seen in the code in this line:

'''python
   network_classifications = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                               "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                               "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
'''

    Then for each Tiny Yolo object in the image the program crops out the bounding rectangle and passes that smaller image to googlenet for a more detailed classification. The program will then display the original frame along with boxes around each detected object and its detailed classification if one was provided by googlenet with sufficient probability.  If GoogLeNet doesn't provide a high enough probability classification then the original Tiny Yolo classification will be used.

There are a few thresholds in the code you may want to tweek if you aren't getting results that you expect:
- <strong>probability_threshold</strong>: This is the minimum probability for boxes to consider as returned from tiny yolo.  This should be between 0.0 and 1.0
- <strong>max_iou</strong>: Dertermines which boxes from Tiny Yolo are actually around the same object based on the intersection-over-union metric.  The closer this is to 1.0 the more similar the boxes need to be to be the same.
- <strong>GOOGLE_PROB_MIN</strong>:  This is the minimum probability from googlenet that will be used to override the general tiny yolo classification with a more specific googlenet classification.  It should be between 0.0 and 1.0

# Makefile
Provided Makefile has various targets that help with the above mentioned tasks.

## make help
Shows available targets

## make all
builds and/or gathers all the required files needed to run the application


## make run_py
Runs the provided python program which shows the camera stream along with the object boxes and classifications

## make clean
Removes all the temporary files that are created by the Makefile
