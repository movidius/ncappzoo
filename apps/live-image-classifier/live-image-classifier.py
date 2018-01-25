#!/usr/bin/python3

# ****************************************************************************
# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.
# ****************************************************************************

# Perform inference on a LIVE camera feed using DNNs on 
# Intel® Movidius™ Neural Compute Stick (NCS)

import mvnc.mvncapi as mvnc
import numpy
import cv2
import os
import sys

# User modifiable input parameters
NCAPPZOO_PATH           = os.path.expanduser( '~/workspace/ncappzoo' )
GRAPH_PATH              = NCAPPZOO_PATH + '/tensorflow/mobilenets/graph'
CATEGORIES_PATH         = NCAPPZOO_PATH + '/tensorflow/mobilenets/categories.txt'
IMAGE_MEAN              = numpy.float16( 127.5 )
IMAGE_STDDEV            = ( 1 / 127.5 )
IMAGE_DIM               = ( 224, 224 )

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

# ---- Step 3: Pre-process the images ----------------------------------------

def pre_process_image():
    # Grab a frame from the camera
    ret, frame = cam.read()
    height, width, channels = frame.shape

    # Extract/crop face and resize it
    x1 = int( width / 3 )
    y1 = int( height / 4 )
    x2 = int( width * 2 / 3 )
    y2 = int( height * 3 / 4 )

    cv2.rectangle( frame, ( x1, y1 ) , ( x2, y2 ), ( 0, 255, 0 ), 2 )
    cv2.imshow( 'NCS real-time inference', frame )
    
    croped_frame = frame[ y1 : y2, x1 : x2 ]
    cv2.imshow( 'Croped frame', croped_frame )

    # resize image [Image size if defined by choosen network, during training]
    croped_frame = cv2.resize( croped_frame, IMAGE_DIM )

    # Mean subtraction & scaling [A common technique used to center the data]
    croped_frame = croped_frame.astype( numpy.float16 )
    croped_frame = ( croped_frame - IMAGE_MEAN ) * IMAGE_STDDEV

    return croped_frame

# ---- Step 4: Offload images, read & print inference results ----------------

def infer_image( graph, img ):

    # Read all categories into a list
    categories = [line.rstrip('\n') for line in 
                   open( CATEGORIES_PATH ) if line != 'classes\n']

    # Load the image as a half-precision floating point array
    graph.LoadTensor( img , 'user object' )

    # Get results from the NCS
    output, userobj = graph.GetResult()

    # Find the index of highest confidence 
    top_prediction = output.argmax()

    # Get execution time
    inference_time = graph.GetGraphOption( mvnc.GraphOption.TIME_TAKEN )

    # Print top prediction
    print( "Prediction: " + str(top_prediction) 
           + " " + categories[top_prediction] 
           + " with %3.1f%% confidence" % (100.0 * output[top_prediction] )
           + " in %.2f ms" % ( numpy.sum( inference_time ) ) )

    return

# ---- Step 5: Unload the graph and close the device -------------------------

def close_ncs_device( device, graph ):
    cam.release()
    cv2.destroyAllWindows()
    graph.DeallocateGraph()
    device.CloseDevice()

# ---- Main function (entry point for this script ) --------------------------

def main():
    device = open_ncs_device()
    graph = load_graph( device )

    while( True ):
        img = pre_process_image()
        infer_image( graph, img )

        if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
            break

    close_ncs_device( device, graph )

# ---- Define 'main' function as the entry point for this script -------------

if __name__ == '__main__':
    main()

# ==== End of file ===========================================================

