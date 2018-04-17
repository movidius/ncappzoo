#!/usr/bin/python3

# ****************************************************************************
# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.
# ****************************************************************************

# DIY smart security camera PoC using Raspberry Pi Camera and 
# Intel® Movidius™ Neural Compute Stick (NCS)

import os
import sys
import numpy
import select
import ntpath
import argparse
import picamera
import picamera.array

import mvnc.mvncapi as mvnc

from PIL import Image
from time import localtime, strftime
from utils import visualize_output
from utils import deserialize_output

# "Class of interest" - Display detections only if they match this class ID
CLASS_PERSON         = 15

# Detection threshold: Minimum confidance to tag as valid detection
CONFIDANCE_THRESHOLD = 0.60 # 60% confidant

# Variable to store commandline arguments
ARGS                 = None

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

def pre_process_image( frame ):

    # Read & resize image
    # [Image size is defined by choosen network, during training]
    img = Image.fromarray( frame )
    img = img.resize( ARGS.dim )
    img = numpy.array( img )

    # Mean subtraction & scaling [A common technique used to center the data]
    img = img.astype( numpy.float16 )
    img = ( img - numpy.float16( ARGS.mean ) ) * ARGS.scale

    return img

# ---- Step 4: Read & print inference results from the NCS -------------------

def infer_image( graph, img, frame ):

    # Load the image as a half-precision floating point array
    graph.LoadTensor( img, 'user object' )

    # Get the results from NCS
    output, userobj = graph.GetResult()

    # Get execution time
    inference_time = graph.GetGraphOption( mvnc.GraphOption.TIME_TAKEN )

    # Deserialize the output into a python dictionary
    output_dict = deserialize_output.ssd( 
                      output, 
                      CONFIDANCE_THRESHOLD, 
                      frame.shape )

    # Print the results (each image/frame may have multiple objects)
    for i in range( 0, output_dict['num_detections'] ):

        # Filter a specific class/category
        if( output_dict.get( 'detection_classes_' + str(i) ) == CLASS_PERSON ):

            cur_time = strftime( "%Y_%m_%d_%H_%M_%S", localtime() )
            print( "Person detected on " + cur_time )

            # Extract top-left & bottom-right coordinates of detected objects 
            (y1, x1) = output_dict.get('detection_boxes_' + str(i))[0]
            (y2, x2) = output_dict.get('detection_boxes_' + str(i))[1]

            # Prep string to overlay on the image
            display_str = ( 
                labels[output_dict.get('detection_classes_' + str(i))]
                + ": "
                + str( output_dict.get('detection_scores_' + str(i) ) )
                + "%" )

            # Overlay bounding boxes, detection class and scores
            frame = visualize_output.draw_bounding_box( 
                        y1, x1, y2, x2, 
                        frame,
                        thickness=4,
                        color=(255, 255, 0),
                        display_str=display_str )

            # Capture snapshots
            img = Image.fromarray( frame )
            photo = ( os.path.dirname(os.path.realpath(__file__))
                      + "/captures/photo_"
                      + cur_time + ".jpg" )
            img.save( photo )

    # If a display is available, show the image on which inference was performed
    if 'DISPLAY' in os.environ:
        img.show()

# ---- Step 5: Unload the graph and close the device -------------------------

def close_ncs_device( device, graph ):
    graph.DeallocateGraph()
    device.CloseDevice()

# ---- Main function (entry point for this script ) --------------------------

def main():

    device = open_ncs_device()
    graph = load_graph( device )

    # Main loop: Capture live stream & send frames to NCS
    with picamera.PiCamera() as camera:
        with picamera.array.PiRGBArray( camera ) as frame:
            while( True ):
                camera.resolution = ( 640, 480 )
                camera.capture( frame, ARGS.colormode, use_video_port=True )
                img = pre_process_image( frame.array )
                infer_image( graph, img, frame.array )

                # Clear PiRGBArray, so you can re-use it for next capture
                frame.seek( 0 )
                frame.truncate()

                # Run the program until <ENTER> is pressed
                i, o, e = select.select( [sys.stdin], [], [], 0.1 )
                if( i ):
                    break

    close_ncs_device( device, graph )

# ---- Define 'main' function as the entry point for this script -------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                         description="DIY smart security camera using \
                         Rapberry Pi Camera and Intel® Movidius™ Neural Compute Stick. \
                         \n Hit <ENTER> to terminate program." )

    parser.add_argument( '-g', '--graph', type=str,
                         default='../../caffe/SSD_MobileNet/graph',
                         help="Absolute path to the neural network graph file." )

    parser.add_argument( '-l', '--labels', type=str,
                         default='../../caffe/SSD_MobileNet/labels.txt',
                         help="Absolute path to labels file." )

    parser.add_argument( '-M', '--mean', type=float,
                         nargs='+',
                         default=[127.5, 127.5, 127.5],
                         help="',' delimited floating point values for image mean." )

    parser.add_argument( '-S', '--scale', type=float,
                         default=0.00789,
                         help="Absolute path to labels file." )

    parser.add_argument( '-D', '--dim', type=int,
                         nargs='+',
                         default=[300, 300],
                         help="Image dimensions. ex. -D 224 224" )

    parser.add_argument( '-c', '--colormode', type=str,
                         default="bgr",
                         help="RGB vs BGR color sequence. This is network dependent." )

    ARGS = parser.parse_args()

    # Load the labels file 
    labels =[ line.rstrip('\n') for line in 
              open( ARGS.labels ) if line != 'classes\n'] 

    main()

# ==== End of file ===========================================================
