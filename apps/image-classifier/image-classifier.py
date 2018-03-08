#!/usr/bin/python3

# ****************************************************************************
# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.
# ****************************************************************************

# How to classify images using DNNs on Intel Neural Compute Stick (NCS)

import os
import sys
import numpy
import ntpath
import argparse
import skimage.io
import skimage.transform

import mvnc.mvncapi as mvnc

# Number of top prodictions to print
NUM_PREDICTIONS		= 6

<<<<<<< HEAD
# User modifiable input parameters
NCAPPZOO_PATH           = '../..'
GRAPH_PATH              = NCAPPZOO_PATH + '/caffe/GoogLeNet/graph' 
IMAGE_PATH              = NCAPPZOO_PATH + '/data/images/cat.jpg'
LABELS_FILE_PATH        = NCAPPZOO_PATH + '/data/ilsvrc12/synset_words.txt'
IMAGE_MEAN              = [ 104.00698793, 116.66876762, 122.67891434]
IMAGE_STDDEV            = 1
IMAGE_DIM               = ( 224, 224 )
=======
# Variable to store commandline arguments
ARGS                    = None
>>>>>>> 7990938... Updated image-classifier to include command line arguments

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

<<<<<<< HEAD
# Mean subtraction & scaling [A common technique used to center the data]
img = img.astype( numpy.float32 )
img = ( img - IMAGE_MEAN ) * IMAGE_STDDEV

# Load the image as a half-precision floating point array
graph.LoadTensor( img.astype( numpy.float16 ), 'user object' )
=======
def pre_process_image():

    # Read & resize image [Image size is defined during training]
    img = skimage.io.imread( ARGS.image )
    img = skimage.transform.resize( img, ARGS.dim, preserve_range=True )

    # Convert RGB to BGR [skimage reads image in RGB, but Caffe uses BGR]
    if( ARGS.colormode == "BGR" ):
        img = img[:, :, ::-1]

    # Mean subtraction & scaling [A common technique used to center the data]
    img = img.astype( numpy.float16 )
    img = ( img - numpy.float16( ARGS.mean ) ) * ARGS.scale

    return img
>>>>>>> 7990938... Updated image-classifier to include command line arguments

# ---- Step 4: Read & print inference results from the NCS -------------------

def infer_image( graph, img ):

    # Load the labels file 
    labels =[ line.rstrip('\n') for line in 
                   open( ARGS.labels ) if line != 'classes\n'] 

    # The first inference takes an additional ~20ms due to memory 
    # initializations, so we make a 'dummy forward pass'.
    graph.LoadTensor( img, 'user object' )
    output, userobj = graph.GetResult()

<<<<<<< HEAD
labels = numpy.loadtxt( LABELS_FILE_PATH, str, delimiter = '\t' )
=======
    # Load the image as a half-precision floating point array
    graph.LoadTensor( img, 'user object' )
>>>>>>> 7990938... Updated image-classifier to include command line arguments

    # Get the results from NCS
    output, userobj = graph.GetResult()

<<<<<<< HEAD
for i in range( 0, 4 ):
	print ('prediction ' + str(i) + ' is ' + labels[order[i]])
=======
    # Sort the indices of top predictions
    order = output.argsort()[::-1][:NUM_PREDICTIONS]

    # Get execution time
    inference_time = graph.GetGraphOption( mvnc.GraphOption.TIME_TAKEN )
>>>>>>> 7990938... Updated image-classifier to include command line arguments

    # Print the results
    print( "\n==============================================================" )
    print( "Top predictions for", ntpath.basename( ARGS.image ) )
    print( "Execution time: " + str( numpy.sum( inference_time ) ) + "ms" )
    print( "--------------------------------------------------------------" )
    for i in range( 0, NUM_PREDICTIONS ):
        print( "%3.1f%%\t" % (100.0 * output[ order[i] ] )
               + labels[ order[i] ] )
    print( "==============================================================" )

    # If a display is available, show the image on which inference was performed
    if 'DISPLAY' in os.environ:
        skimage.io.imshow( ARGS.image )
        skimage.io.show()

# ---- Step 5: Unload the graph and close the device -------------------------

def close_ncs_device( device, graph ):
    graph.DeallocateGraph()
    device.CloseDevice()

# ---- Main function (entry point for this script ) --------------------------

def main():

    device = open_ncs_device()
    graph = load_graph( device )

    img = pre_process_image()
    infer_image( graph, img )

    close_ncs_device( device, graph )

# ---- Define 'main' function as the entry point for this script -------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                         description="Image classifier using \
                         Intel® Movidius™ Neural Compute Stick." )

    parser.add_argument( '-g', '--graph', type=str,
                         default='../../caffe/GoogLeNet/graph',
                         help="Absolute path to the neural network graph file." )

    parser.add_argument( '-i', '--image', type=str,
                         default='../../data/images/cat.jpg',
                         help="Absolute path to the image that needs to be inferred." )

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
