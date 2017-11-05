#!/usr/bin/python3

# ****************************************************************************
# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.
# ****************************************************************************

# How to sequentially classify multiple images using DNNs on 
# Intel® Movidius™ Neural Compute Stick (NCS)

import mvnc.mvncapi as mvnc
import numpy
import cv2
import os
from os import listdir, path
from os.path import expanduser, isfile, join
import timeit

# User modifiable input parameters
NCAPPZOO_PATH           = expanduser( '~/workspace/ncappzoo' )
IMAGES_PATH             = NCAPPZOO_PATH + '/data/images'
LABELS_FILE_PATH        = NCAPPZOO_PATH + '/data/ilsvrc12/synset_words.txt'
IMAGE_MEANS_FILE_PATH   = NCAPPZOO_PATH + '/data/ilsvrc12/ilsvrc_2012_mean.npy'
GRAPH_PATH              = NCAPPZOO_PATH + '/caffe/GoogLeNet/graph'
IMAGE_DIM               = ( 224, 224 )

# ---- Step 1: Open the enumerated device and get a handle to it -------------

# Look for enumerated NCS device(s); quit program if none found.
devices = mvnc.EnumerateDevices()
if len( devices ) == 0:
	print( 'No devices found' )
	quit()

# Get a handle to the first enumerated device and open it
device = mvnc.Device( devices[0] )
device.OpenDevice()

# ---- Step 2: Load a graph file onto the NCS device -------------------------

# Read the graph file into a buffer
with open( GRAPH_PATH, mode='rb' ) as f:
	blob = f.read()

# Load the graph buffer into the NCS
graph = device.AllocateGraph( blob )

# ---- Step 3: Pre-process the images ----------------------------------------

# Load the labels file corresponsding to ilsvrc12 dataset
labels = numpy.loadtxt( LABELS_FILE_PATH, str, delimiter = '\t' )

# Load the mean file [This file was downloaded from ilsvrc website]
ilsvrc_mean = numpy.load( IMAGE_MEANS_FILE_PATH ).mean( 1 ).mean( 1 )

# Read & pre-process all images in the folder [Image size is defined during training]
imgarray = []
print_imgarray = []

onlyfiles = [ f for f in listdir(IMAGES_PATH) if isfile( join( IMAGES_PATH, f ) ) ]

for file in onlyfiles:
    fimg = IMAGES_PATH + "/" + file
#    print( "Opening file ", fimg )
    img = cv2.imread( fimg )
    print_imgarray.append( cv2.resize( img, ( 700, 700 ) ) )
    img = cv2.resize( img, IMAGE_DIM )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Mean subtraction [A common technique used to center the data]
    img = img.astype( numpy.float32 )
    img[:,:,0] = (img[:,:,0] - ilsvrc_mean[0])
    img[:,:,1] = (img[:,:,1] - ilsvrc_mean[1])
    img[:,:,2] = (img[:,:,2] - ilsvrc_mean[2])

    img = img.astype( numpy.float16 )
    imgarray.append( img )

# ---- Step 4: Read & print inference results from the NCS -------------------
for index, img in enumerate( imgarray ):
    # Load the image as a half-precision floating point array
    graph.LoadTensor( img , 'user object' )

    # Get the results from NCS
    output, userobj = graph.GetResult()
    order = output.argsort()[::-1][:6]

    # Print prediction results on the terminal window
    print( labels[order[0]] )

    # Display inferred image with top pridiction
    cv2.putText( print_imgarray[index], labels[order[0]], 
                 ( 10,30 ), cv2.FONT_HERSHEY_SIMPLEX, 1, ( 0, 255, 0 ), 2 )

    cv2.imshow( 'Image Classifier', print_imgarray[index] )

    cv2.waitKey( 1 )

# ---- Step 5: Unload the graph and close the device -------------------------
cv2.waitKey( 0 )
graph.DeallocateGraph()
device.CloseDevice()

# ==== End of file ===========================================================

