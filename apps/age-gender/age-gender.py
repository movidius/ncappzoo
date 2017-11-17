#!/usr/bin/python3

# ****************************************************************************
# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.
# ****************************************************************************

# How to predict a person's age and gender using OpenCV, Age/Gender CNN and
# Intel® Movidius™ Neural Compute Stick (NCS)

import mvnc.mvncapi as mvnc
import numpy
import cv2
import os
import sys

# User modifiable input parameters for NCS
NCAPPZOO_PATH           = os.path.expanduser( '~/workspace/ncappzoo' )
GRAPH_PATH              = NCAPPZOO_PATH + '/caffe/AgeNet/graph'
IMAGE_MEAN_PATH         = NCAPPZOO_PATH + '/data/age_gender/age_gender_mean.npy'
LABELS_AGE              = [ '0-2','4-6','8-12','15-20','25-32','38-43','48-53','60-100' ]
LABELS_GENDER           = [ 'Male', 'Female' ]
IMAGE_STDDEV            = 1
IMAGE_DIM               = ( 227, 227 )

VIDEO_INDEX             = 0
cam                     = cv2.VideoCapture( VIDEO_INDEX )

# ---- Step 1: Open the enumerated device and get a handle to it -------------

def open_ncs_device():

    # Look for enumerated NCS device(s); quit program if none found.
    devices = mvnc.EnumerateDevices()
    if len( devices ) == 0:
        print( 'No devices found' )
        quit()

    # Get a handle to the first enumerated device and open it
    device = mvnc.Device( devices[0] )
    device.OpenDevice()
    return device

# ---- Step 2: Load a graph file onto the NCS device -------------------------

def load_graph( device ):

    # Read the graph file into a buffer
    with open( GRAPH_PATH, mode='rb' ) as f:
        blob = f.read()

    # Load the graph buffer into the NCS
    graph = device.AllocateGraph( blob )

    return graph

# ---- Step 3: Pre-process the image -----------------------------------------

def pre_process_image():
    # Grab a frame from the camera
    ret, frame = cam.read()
    height, width, channels = frame.shape
    image_mean = numpy.float16( numpy.load( IMAGE_MEAN_PATH ).mean( 1 ).mean( 1 ) )

    # Extract/crop face and resize it [A common technique used to center the data]
    x1 = int( width / 3 )
    y1 = int( height / 4 )
    x2 = int( width * 2 / 3 )
    y2 = int( height * 3 / 4 )

    cv2.rectangle( frame, ( x1, y1 ) , ( x2, y2 ), ( 0, 255, 0 ), 2 )
    cv2.imshow( 'Age/Gender', frame )
    
    croped_frame = frame[ y1 : y2, x1 : x2 ]
    cv2.imshow( 'Croped face', croped_frame )

    # Uncomment this line if you want to use a static image instead
    # croped_frame = cv2.imread( "./image.jpg" )
    croped_frame = cv2.resize( croped_frame, IMAGE_DIM )

    # Convert RGB to BGR [skimage reads image in RGB, but Caffe uses BGR]
    # croped_frame = croped_frame[:, :, ::-1] 

    # Mean subtraction & scaling [A common technique used to center the data]
    croped_frame = croped_frame.astype( numpy.float16 )
    croped_frame = ( croped_frame - image_mean ) * IMAGE_STDDEV

    return croped_frame

# ---- Step 4: Offload image onto the NCS for inference ----------------------

def infer_image( graph, img ):
    # Load the image as a half-precision floating point array
    graph.LoadTensor( img , 'user object' )

    # Get the results from NCS
    output, userobj = graph.GetResult()

    # Print the results
    top_prediction = output.argmax()

    # Display inferred image with top pridiction
    print( "Age: " + LABELS_AGE[top_prediction] + 
            " with %3.1f%% confidence" % (100.0 * output[top_prediction] ) )

# ---- Step 5: Unload the graph and close the device -------------------------

def close_ncs_device( device, graph ):
    cam.release()
    cv2.destroyAllWindows()
    graph.DeallocateGraph()
    device.CloseDevice()

def main():
    device = open_ncs_device()
    graph = load_graph( device )

    while( True ):
        img = pre_process_image()
        infer_image( graph, img )

        if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
            break

    close_ncs_device( device, graph )

if __name__ == '__main__':
    main()

# ==== End of file ===========================================================

