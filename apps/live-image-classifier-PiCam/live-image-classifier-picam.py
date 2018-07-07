#!/usr/bin/python3

# ****************************************************************************
# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.
# ****************************************************************************

# Perform inference on a LIVE camera feed using DNNs on 
# Intel® Movidius™ Neural Compute Stick (NCS)

import os
import cv2
import sys
import numpy
import ntpath
import argparse
import skimage.io
import skimage.transform

import mvnc.mvncapi as mvnc

import picamera
from picamera.array import PiRGBArray

# Variable to store commandline arguments
ARGS                    = None

# PiCam objects
camera = None
rawCapture = None

# ---- Step 1: Open the enumerated device and get a handle to it -------------

def open_ncs_device():

    # Look for enumerated NCS device(s); quit program if none found.
    devices = mvnc.EnumerateDevices()
    if len( devices ) == 0:
        print( "No devices found" )
        quit()

    # Get a handle to the first enumerated device and open it
    device = mvnc.Device( devices[0] )
    device.OpenDevice()

    return device

# ---- Step 2: Load a graph file onto the NCS device -------------------------

def load_graph( device ):

    # Read the graph file into a buffer
    with open( ARGS.graph, mode='rb' ) as f:
        blob = f.read()

    # Load the graph buffer into the NCS
    graph = device.AllocateGraph( blob )

    return graph

# ---- Step 3: Pre-process the images ----------------------------------------

def pre_process_image(frame):

    # Grab a frame from the camera
    height, width, channels = frame.shape

    # Extract/crop a section of the frame and resize it
    x1 = int( width / 3 )
    y1 = int( height / 4 )
    x2 = int( width * 2 / 3 )
    y2 = int( height * 3 / 4 )

    cv2.rectangle( frame, ( x1, y1 ) , ( x2, y2 ), ( 0, 255, 0 ), 2 )
    img = frame[ y1 : y2, x1 : x2 ]

    # Resize image [Image size if defined by choosen network, during training]
    img = cv2.resize( img, tuple( ARGS.dim ) )

    # Convert RGB to BGR [skimage reads image in RGB, but Caffe uses BGR]
    if( ARGS.colormode == "BGR" ):
        img = img[:, :, ::-1]

    # Mean subtraction & scaling [A common technique used to center the data]
    img = img.astype( numpy.float16 )
    img = ( img - numpy.float16( ARGS.mean ) ) * ARGS.scale

    return img

# ---- Step 4: Read & print inference results from the NCS -------------------

def infer_image( graph, img, frame):

    # Load the labels file 
    labels =[ line.rstrip('\n') for line in 
                   open( ARGS.labels ) if line != 'classes\n'] 

    # Load the image as a half-precision floating point array
    graph.LoadTensor( img, 'user object' )

    # Get the results from NCS
    output, userobj = graph.GetResult()

    # Find the index of highest confidence 
    top_prediction = output.argmax()

    # Get execution time
    inference_time = graph.GetGraphOption( mvnc.GraphOption.TIME_TAKEN )

    print(  "I am %3.1f%%" % (100.0 * output[top_prediction] ) + " confidant"
            + " you are " + labels[top_prediction]
            + " ( %.2f ms )" % ( numpy.sum( inference_time ) ) )

    # If a display is available, show the image on which inference was performed
    if 'DISPLAY' in os.environ:
        cv2.imshow( 'NCS live inference', frame )

# ---- Step 5: Unload the graph and close the device -------------------------

def close_ncs_device( device, graph ):
    graph.DeallocateGraph()
    device.CloseDevice()
    cam.release()
    cv2.destroyAllWindows()

# ---- Main function (entry point for this script ) --------------------------

def main():

    device = open_ncs_device()
    graph = load_graph( device )

    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grap the numpy array representing the image, then initialize the timestamp and occupied/unoccupied text
        image = frame.array    
        
        img = pre_process_image(image)
        infer_image( graph, img, image )

        # Display the frame for 5ms, and close the window so that the next frame 
        # can be displayed. Close the window if 'q' or 'Q' is pressed.
        if( cv2.waitKey( 1 ) & 0xFF == ord( 'q' ) ):
            break
        
        # clear the stream in preparation for the next frame 
        rawCapture.truncate(0)

    close_ncs_device( device, graph )

# ---- Define 'main' function as the entry point for this script -------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                         description="Image classifier using \
                         Intel® Movidius™ Neural Compute Stick." )

    parser.add_argument( '-g', '--graph', type=str,
                         default='../../caffe/GenderNet/graph',
                         help="Absolute path to the neural network graph file." )

    parser.add_argument( '-l', '--labels', type=str,
                         default='../../data/age_gender/gender_categories.txt',
                         help="Absolute path to labels file." )

    parser.add_argument( '-M', '--mean', type=float,
                         nargs='+',
                         default=[78.42633776, 87.76891437, 114.89584775],
                         help="',' delimited floating point values for image mean." )

    parser.add_argument( '-S', '--scale', type=float,
                         default=1,
                         help="Absolute path to labels file." )

    parser.add_argument( '-D', '--dim', type=int,
                         nargs='+',
                         default=[227, 227],
                         help="Image dimensions. ex. -D 224 224" )

    parser.add_argument( '-c', '--colormode', type=str,
                         default="RGB",
                         help="RGB vs BGR color sequence. \
                               Defined during model training." )


    ARGS = parser.parse_args()
    
    #initialize the camera and grap a referance to the raw camera capture
    camera = picamera.PiCamera()
    camera.resolution = (720,576) #default resolution
    camera.vflip = True

    rawCapture = PiRGBArray(camera)


    main()

# ==== End of file ===========================================================
