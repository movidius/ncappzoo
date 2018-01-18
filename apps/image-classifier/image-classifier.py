#!/usr/bin/python3

# ****************************************************************************
# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.
# ****************************************************************************

# How to classify images using DNNs on Intel Neural Compute Stick (NCS)

import mvnc.mvncapi as mvnc
import skimage
from skimage import io, transform
import numpy
import os
import sys

# User modifiable input parameters
NCAPPZOO_PATH           = '../..'
GRAPH_PATH              = NCAPPZOO_PATH + '/caffe/GoogLeNet/graph'
IMAGE_PATH              = NCAPPZOO_PATH + '/data/images/cat.jpg'
CATEGORIES_PATH         = NCAPPZOO_PATH + '/data/ilsvrc12/synset_words.txt'
IMAGE_MEAN              = numpy.float16( [104.00698793, 116.66876762, 122.67891434] )
IMAGE_STDDEV            = ( 1 )
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

# ---- Step 3: Offload image onto the NCS to run inference -------------------

# Read & resize image [Image size is defined during training]
img = print_img = skimage.io.imread( IMAGE_PATH )
img = skimage.transform.resize( img, IMAGE_DIM, preserve_range=True )

# Convert RGB to BGR [skimage reads image in RGB, but Caffe uses BGR]
img = img[:, :, ::-1]

# Mean subtraction & scaling [A common technique used to center the data]
img = img.astype( numpy.float16 )
img = ( img - IMAGE_MEAN ) * IMAGE_STDDEV

# Load the image as a half-precision floating point array
graph.LoadTensor( img, 'user object' )

# ---- Step 4: Read & print inference results from the NCS -------------------

# Get the results from NCS
output, userobj = graph.GetResult()

# Print the results
print('\n------- predictions --------')

# Read all categories into a list
categories = [line.rstrip('\n') for line in
              open( CATEGORIES_PATH ) if line != 'classes\n']

order = output.argsort()[::-1][:6]

# Get execution time
inference_time = graph.GetGraphOption( mvnc.GraphOption.TIME_TAKEN )

for i in range( 0, 4 ):
    print( "Prediction for "
            + ": " + categories[order[i]]
            + " with %3.1f%% confidence"
            % (100.0 * output[order[i]] )
            + " in %.2f ms" % ( numpy.sum( inference_time ) ) )

# If a display is available, show the image on which inference was performed
if 'DISPLAY' in os.environ:
    skimage.io.imshow( IMAGE_PATH )
    skimage.io.show( )

# ---- Step 5: Unload the graph and close the device -------------------------

graph.DeallocateGraph()
device.CloseDevice()

# ==== End of file ===========================================================

