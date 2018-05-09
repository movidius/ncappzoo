#!/usr/bin/python3

# ****************************************************************************
# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.
# ****************************************************************************

# How to sequentially classify multiple images using DNNs on 
# Intel® Movidius™ Neural Compute Stick (NCS)

import os
import sys
import glob
import numpy
import ntpath
import argparse
import skimage.io
import skimage.transform

import mvnc.mvncapi as mvnc

# Max number of images to process. 
# Avoid exhausting the memory with 1000s of images
MAX_IMAGE_COUNT		= 200

# Variable to store commandline arguments
ARGS                = None

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

def pre_process_image():

    img_list = []

    print( "\n\nPre-processing images..." )

    # Create a list of all files in current directory & sub-directories
    img_paths = [ y for x in os.walk( ARGS.image ) 
                  for y in glob.glob( os.path.join( x[0], '*.jpg' ) ) ]

    # Sort file names in alphabetical order
    img_paths.sort()

    for img_index, img_name in enumerate( img_paths ):

        # Set a limit on the image count, so that it doesn't fill up the memory
        if img_index >= MAX_IMAGE_COUNT:
            break

        # Read & resize image [Image size is defined during training]
        img = skimage.io.imread( img_name )
        img = skimage.transform.resize( img, ARGS.dim, preserve_range=True )

        # Convert RGB to BGR [skimage reads image in RGB, but Caffe uses BGR]
        if( ARGS.colormode == "BGR" ):
            img = img[:, :, ::-1]

        # Mean subtraction & scaling [A common technique used to center the data]
        img = ( img - ARGS.mean) * ARGS.scale

        img_list.append( img )

    return img_list, img_paths

# ---- Step 4: Read & print inference results from the NCS -------------------

def infer_image( graph, img_list, img_paths, fifo_in, fifo_out ):

    # Load the labels file 
    labels =[ line.rstrip('\n') for line in 
                   open( ARGS.labels ) if line != 'classes\n'] 

    print( "\n==============================================================" )
    for img_index, img in enumerate( img_list ):
        # Load the image as an array
        graph.queue_inference_with_fifo_elem( fifo_in, fifo_out, img.astype(numpy.float32), None )
        # Get the results from NCS
        output, userobj = fifo_out.read_elem()

        # Get execution time
        inference_time = graph.get_option( mvnc.GraphOption.RO_TIME_TAKEN )

        # Find the index of highest confidence 
        top_prediction = output.argmax()

        # Print top predictions for each image
        print(  "Predicted " + ntpath.basename( img_paths[img_index] )
                + " as " + labels[top_prediction]
                + " in %.2f ms" % ( numpy.sum( inference_time ) ) 
                + " with %3.1f%%" % (100.0 * output[top_prediction] )
                + " confidence." )

    print( "==============================================================\n" )

# ---- Step 5: Close/clean up fifos, graph, and device -------------------------

def clean_up(device, graph, fifo_in, fifo_out):
    fifo_in.destroy()
    fifo_out.destroy()
    graph.destroy()
    device.close()
    device.destroy()

# ---- Main function (entry point for this script ) --------------------------

def main():

    device = open_ncs_device()
    graph, fifo_in, fifo_out = load_graph( device )

    img_list, img_paths = pre_process_image()
    infer_image( graph, img_list, img_paths, fifo_in, fifo_out )

    clean_up(device, graph, fifo_in, fifo_out)

# ---- Define 'main' function as the entry point for this script -------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                         description="Image classifier using \
                         Intel® Movidius™ Neural Compute Stick." )

    parser.add_argument( '-g', '--graph', type=str,
                         default='../../caffe/GoogLeNet/graph',
                         help="Absolute path to the neural network graph file." )

    parser.add_argument( '-i', '--image', type=str,
                         default='../../data/images',
                         help="Absolute path to the folder where all images are stored." )

    parser.add_argument( '-l', '--labels', type=str,
                         default='../../data/ilsvrc12/synset_words.txt',
                         help="Absolute path to labels file." )

    parser.add_argument( '-M', '--mean', type=float,
                         nargs='+',
                         default=[104.00698793, 116.66876762, 122.67891434],
                         help="',' delimited floating point values for image mean." )

    parser.add_argument( '-S', '--scale', type=float,
                         default=1,
                         help="Absolute path to labels file." )

    parser.add_argument( '-D', '--dim', type=int,
                         nargs='+',
                         default=[224, 224],
                         help="Image dimensions. ex. -D 224 224" )

    parser.add_argument( '-c', '--colormode', type=str,
                         default="BGR",
                         help="RGB vs BGR color sequence. TensorFlow = RGB, Caffe = BGR" )


    ARGS = parser.parse_args()

    main()

# ==== End of file ===========================================================
