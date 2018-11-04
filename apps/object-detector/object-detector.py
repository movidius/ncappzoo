#!/usr/bin/python3

# ****************************************************************************
# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.
# ****************************************************************************

# How to run Single Shot Multibox Detectors (SSD)
# on Intel® Movidius™ Neural Compute Stick (NCS)

import os
import sys
import numpy as np
import ntpath
import argparse
import skimage.io
import skimage.transform

import mvnc.mvncapi as mvnc

from utils import visualize_output
from utils import deserialize_output

# Detection threshold: Minimum confidance to tag as valid detection
CONFIDANCE_THRESHOLD = 0.60 # 60% confidant

# Variable to store commandline arguments
ARGS                 = None

# ---- Step 1: Open the enumerated device and get a handle to it -------------

def open_ncs_device():

    # Look for enumerated NCS device(s); quit program if none found.
    devices = mvnc.enumerate_devices()
    if len( devices ) == 0:
        print( "No devices found" )
        quit()

    # Get a handle to the first enumerated device and open it
    device = mvnc.Device( devices[0] )
    device.open()

    return device

# ---- Step 2: Load a graph file onto the NCS device -------------------------

def load_graph( device ):

    # Read the graph file into a buffer
    with open( ARGS.graph, mode='rb' ) as f:
        blob = f.read()

    # Load the graph buffer into the NCS
    graph = mvnc.Graph( ARGS.graph )
    # Set up fifos
    fifo_in, fifo_out = graph.allocate_with_fifos( device, blob )

    return graph, fifo_in, fifo_out

# ---- Step 3: Pre-process the images ----------------------------------------

def pre_process_image( img_original ):

    # Read & Resize image [Image size is defined during training]
    img = skimage.transform.resize( img_original, ARGS.dim, preserve_range=True )

    # Convert RGB to BGR [skimage reads image in RGB, some networks may need BGR]
    if( ARGS.colormode == "BGR" ):
        img = img[:, :, ::-1]

    # Mean subtraction & scaling [A common technique used to center the data]
    img = ( img - ARGS.mean ) * ARGS.scale

    return img

# ---- Step 4: Read & print inference results from the NCS -------------------

def infer_image( graph, img_original, img_preprocessed, fifo_in, fifo_out ):

    # The first inference takes an additional ~20ms due to memory
    # initializations, so we make a 'dummy forward pass'.
    graph.queue_inference_with_fifo_elem( fifo_in, fifo_out, img_preprocessed.astype(np.float32), None )
    output, userobj = fifo_out.read_elem()

    # Load the image as an array
    graph.queue_inference_with_fifo_elem( fifo_in, fifo_out, img_preprocessed.astype(np.float32), None )
    output, userobj = fifo_out.read_elem()

    # Get execution time
    inference_time = graph.get_option( mvnc.GraphOption.RO_TIME_TAKEN )

    # Deserialize the output into a python dictionary
    if ARGS.network == 'SSD':
        output_dict = deserialize_output.ssd( output, CONFIDANCE_THRESHOLD, img_original.shape )
    elif ARGS.network == 'TinyYolo':
        output_dict = deserialize_output.tinyyolo( output, CONFIDANCE_THRESHOLD, img_original.shape )

    # Print the results
    print( "\n==============================================================" )
    print( "I found these objects in", ntpath.basename( ARGS.image ) )
    print( "Execution time: " + str( np.sum( inference_time ) ) + "ms" )
    print( "--------------------------------------------------------------" )
    for i in range( 0, output_dict['num_detections'] ):
        print( "%3.1f%%\t" % output_dict['detection_scores_' + str(i)]
               + labels[ int(output_dict['detection_classes_' + str(i)]) ]
               + ": Top Left: " + str( output_dict['detection_boxes_' + str(i)][0] )
               + " Bottom Right: " + str( output_dict['detection_boxes_' + str(i)][1] ) )

        # Draw bounding boxes around valid detections
        (y1, x1) = output_dict.get('detection_boxes_' + str(i))[0]
        (y2, x2) = output_dict.get('detection_boxes_' + str(i))[1]

        # Prep string to overlay on the image
        display_str = (
                labels[output_dict.get('detection_classes_' + str(i))]
                + ": "
                + str( output_dict.get('detection_scores_' + str(i) ) )
                + "%" )

        img_draw = visualize_output.draw_bounding_box(
                       y1, x1, y2, x2,
                       img_original,
                       thickness=4,
                       color=(255, 255, 0),
                       display_str=display_str )

    print( "==============================================================\n" )

    # If a display is available, show the image on which inference was performed
    if 'DISPLAY' in os.environ:
        skimage.io.imshow( img_draw )
        skimage.io.show()

# ---- Step 5: Unload the graph and close the device -------------------------

def clean_up( device, graph, fifo_in, fifo_out ):
    fifo_in.destroy()
    fifo_out.destroy()
    graph.destroy()
    device.close()
    device.destroy()

# ---- Main function (entry point for this script ) --------------------------

def main():

    device = open_ncs_device()
    graph, fifo_in, fifo_out = load_graph( device )

    img_original = skimage.io.imread( ARGS.image )
    img_preprocessed = pre_process_image( img_original )
    infer_image( graph, img_original, img_preprocessed, fifo_in, fifo_out )

    clean_up( device, graph, fifo_in, fifo_out )

# ---- Define 'main' function as the entry point for this script -------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                         description="Object detection using SSD on \
                         Intel® Movidius™ Neural Compute Stick." )

    parser.add_argument( '-n', '--network', type=str,
                         default='SSD',
                         help="network name: SSD or TinyYolo." )

    parser.add_argument( '-g', '--graph', type=str,
                         default='../../caffe/SSD_MobileNet/graph',
                         help="Absolute path to the neural network graph file." )

    parser.add_argument( '-i', '--image', type=str,
                         default='../../data/images/nps_chair.png',
                         help="Absolute path to the image that needs to be inferred." )

    parser.add_argument( '-l', '--labels', type=str,
                         default='./labels.txt',
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
