#!/usr/bin/python3

# ****************************************************************************
# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.
# ****************************************************************************

# How to sequentially classify multiple images using DNNs on 
# Intel® Movidius™ Neural Compute Stick (NCS)

import mvnc.mvncapi as mvnc
import numpy
import skimage
from skimage import io, transform 
import os
from os import listdir, path
from os.path import expanduser, isfile, join

# User modifiable input parameters
NCAPPZOO_PATH           = expanduser( '~/workspace/ncappzoo' )
GRAPH_PATH              = NCAPPZOO_PATH + '/tensorflow/mobilenets/graph'
IMAGES_PATH             = NCAPPZOO_PATH + '/data/images'
LABELS_PATH             = NCAPPZOO_PATH + '/tensorflow/mobilenets/categories.txt'
IMAGE_MEAN              = numpy.float16( 127.5 )
IMAGE_STDDEV            = ( 1 / 127.5 )
IMAGE_DIM               = ( 224, 224 )

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

    # Read all images in the folder
    imgarray = []
    print_imgarray = []

    onlyfiles = [ f for f in listdir(IMAGES_PATH) 
                  if isfile( join( IMAGES_PATH, f ) ) ]

    print( "\n\nPre-processing images..." )

    for file in onlyfiles:
        fimg = IMAGES_PATH + "/" + file
#        print( "Opening file ", fimg )

        # Read & resize image [Image size is defined during training]
        img = skimage.io.imread( fimg )
        print_imgarray.append( skimage.transform.resize( img, ( 700, 700 ) ) ) 
        img = skimage.transform.resize( img, IMAGE_DIM, preserve_range=True )

        # Convert RGB to BGR [skimage reads image in RGB, but Caffe uses BGR]
        img = img[:, :, ::-1]

        # Mean subtraction & scaling [A common technique used to center the data]
        img = img.astype( numpy.float16 )
        img = ( img - IMAGE_MEAN ) * IMAGE_STDDEV

        imgarray.append( img )

    return imgarray, print_imgarray

# ---- Step 4: Offload images, read & print inference results ----------------

def infer_image( graph, imgarray, print_imgarray ):

    # Load the labels file 
    labels = numpy.loadtxt( LABELS_PATH, str, delimiter = '\t' )

    print( "\n---- Predictions ----" )

    for index, img in enumerate( imgarray ):
        # Load the image as a half-precision floating point array
        graph.LoadTensor( img , 'user object' )

        # Get the results from NCS
        output, userobj = graph.GetResult()
        order = output.argsort()[::-1][:6]

        # Print prediction results on the terminal window
        print( labels[order[0] + 1] )

        # Display the image on which inference was performed
#        if 'DISPLAY' in os.environ:
#            skimage.io.imshow( print_imgarray[index] )
#            skimage.io.show( )

# ---- Step 5: Unload the graph and close the device -------------------------

def close_ncs_device( device, graph ):
    graph.DeallocateGraph()
    device.CloseDevice()

def main():
    device = open_ncs_device()
    graph = load_graph( device )

    imgarray, print_imgarray = pre_process_image()
    infer_image( graph, imgarray, print_imgarray )

    close_ncs_device( device, graph )

if __name__ == '__main__':
    main()

# ==== End of file ===========================================================

