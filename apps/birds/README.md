# Birds
## Introduction
The birds application is a bird recognition and classification program.  It uses both the NCS along with TinyYolo and GoogLeNet to take an image and first get bounding boxes for birds in the image and then uses GoogLeNet to further classify the birds found.  It depends on the caffe/TinyYolo and caffe/GoogLeNet appzoo examples.
The provided Makefile does the following:
1. Builds both TinyYolo and GoogLeNet from their repective directories within the repo.
2. Copies the built NCS graph files from TinyYolo and GoogLeNet to the current directory.
3. Downloads some images for the program to use
4. Runs the provided birds.py program that creates a GUI window which cycles through all the .jpg images in the current directory and shows the birds and all their classifications using the networks using the Inference Engine API

## How to run the sample
To run the sample, change directory to the birds application folder and use the command: ```make run```


## How it Works
One NCS device is needed to run this application.  For each image in the directory the program first runs a tiny yolo inference to find all birds in the image.  Then for each bird in the image the program crops out the bounding rectangle and passes that smaller image to googlenet for a more detailed classification (ie bald eagle, etc.)  The program will then display the original image along with boxes around each detected bird and its detailed classification if one was provided.

There are a few thresholds in the code you may want to tweek if you aren't getting results that you expect when you use your own images.
- <strong>DETECTION_THRESHOLD</strong>: This is the minimum probability for boxes to consider as returned from tiny yolo.  This should be between 0.0 and 1.0. This variable is found in the tiny_yolo_processor.py script.
- <strong>MAX_IOU</strong>: Dertermines which boxes from Tiny Yolo are actually around the same object based on the intersection-over-union metric.  The closer this is to 1.0 the more similar the boxes need to be to be the same. This variable is found in the tiny_yolo_processor.py script.
- <strong>GOOGLE_PROB_MIN</strong>:  This is the minimum probability from googlenet that will be used to override the general tiny yolo classification with a more specific googlenet classification.  It should be between 0.0 and 1.0. This variable is found in the birds.py script.

## Makefile
Provided Makefile has various targets that help with the above mentioned tasks.

### make help
Shows available targets

### make all
builds and/or gathers all the required files needed to run the application

### make run_py
Runs the provided birds.py python program which does the bird classification

### make clean
Removes all the temporary files that are created by the Makefile
