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
from glob import glob
import ntpath

# User modifiable input parameters
NCAPPZOO_PATH           = '../..' 
GRAPH_PATH              = NCAPPZOO_PATH + '/tensorflow/mobilenets/graph'
IMAGE_PATH              = NCAPPZOO_PATH + '/data/images'
CATEGORIES_PATH         = NCAPPZOO_PATH + '/tensorflow/mobilenets/categories.txt'
IMAGE_MEAN              = numpy.float16( 127.5 )
IMAGE_STDDEV            = ( 1 / 127.5 )
IMAGE_DIM               = ( 224, 224 )

# Max number of images to process
MAX_IMAGE_COUNT         = 200

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

    print( "\n\nPre-processing images..." )

    # Create a list of all files in current directory & sub-directories
    file_list = [ y for x in os.walk( IMAGE_PATH ) 
                  for y in glob( os.path.join( x[0], '*.jpg' ) ) ]

    for file_index, file_name in enumerate( file_list ):

        # Set a limit on the image count, so that it doesn't fill up the memory
        if file_index >= MAX_IMAGE_COUNT:
            break

        # print( "Opening file ", file_name )

        # Read & resize image [Image size is defined during training]
        img = skimage.io.imread( file_name )
        print_imgarray.append( skimage.transform.resize( img, ( 700, 700 ) ) ) 
        img = skimage.transform.resize( img, IMAGE_DIM, preserve_range=True )

        # Convert RGB to BGR [skimage reads image in RGB, but Caffe uses BGR]
        img = img[:, :, ::-1]

        # Mean subtraction & scaling [A common technique used to center the data]
        img = img.astype( numpy.float16 )
        img = ( img - IMAGE_MEAN ) * IMAGE_STDDEV

        imgarray.append( img )

    return file_list, imgarray, print_imgarray

# ---- Step 4: Offload images, read & print inference results ----------------

def infer_image( graph, file_list, imgarray, print_imgarray ):

    # Load the labels file 
    labels =[ line.rstrip('\n') for line in 
                   open( CATEGORIES_PATH ) if line != 'classes\n'] 

    for index, img in enumerate( imgarray ):
        # Load the image as a half-precision floating point array
        graph.LoadTensor( img , 'user object' )

        # Get the results from NCS
        output, userobj = graph.GetResult()

        # Get execution time
        inference_time = graph.GetGraphOption( mvnc.GraphOption.TIME_TAKEN )

        # Find the index of highest confidence 
        top_prediction = output.argmax()

        # Print top prediction
        print( "Prediction for " 
                + ntpath.basename( file_list[index] ) 
                + ": " + labels[top_prediction] 
                + " with %3.1f%% confidence" 
                % (100.0 * output[top_prediction] )
                + " in %.2f ms" % ( numpy.sum( inference_time ) ) )

        # Display the image on which inference was performed
        # ---------------------------------------------------------
        # Uncomment below line if you get 
        #  'No suitable plugin registered for imshow' error message
        #skimage.io.use_plugin( 'matplotlib' )
        # ---------------------------------------------------------
        #skimage.io.imshow( print_imgarray[index] )
        #skimage.io.show( )

# ---- Step 5: Unload the graph and close the device -------------------------

def close_ncs_device( device, graph ):
    graph.DeallocateGraph()
    device.CloseDevice()

# ---- Main function (entry point for this script ) --------------------------

def main():

    device = open_ncs_device()
    graph = load_graph( device )

    file_list, imgarray, print_imgarray = pre_process_image()
    infer_image( graph, file_list, imgarray, print_imgarray )

    close_ncs_device( device, graph )

# ---- Define 'main' function as the entry point for this script -------------

if __name__ == '__main__':
    main()

# ==== End of file ===========================================================

