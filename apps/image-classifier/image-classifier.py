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
NCAPPZOO_PATH           = os.path.expanduser( '~/workspace/ncappzoo' )
GRAPH_PATH              = NCAPPZOO_PATH + '/caffe/GoogLeNet/GoogLeNet.graph'
IMAGES_PATH             = sys.argv[1] 
LABELS_FILE_PATH        = NCAPPZOO_PATH + '/data/ilsvrc12/synset_words.txt'
IMAGE_MEANS_FILE_PATH   = NCAPPZOO_PATH + '/data/ilsvrc12/ilsvrc_2012_mean.npy'
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

# Load the mean file [This file was downloaded from ilsvrc website]
ilsvrc_mean = numpy.load( IMAGE_MEANS_FILE_PATH ).mean( 1 ).mean( 1 )

# Read image into an ndarray
img = print_img = skimage.io.imread( IMAGES_PATH )

# Resize the image [ Image size if defined during training ]
img = skimage.transform.resize( img, IMAGE_DIM, preserve_range=True )

# Convert RGB to BGR [skimage reads image in RGB, but Caffe uses BGR]
img = img[:, :, ::-1]

# Mean subtraction [A common technique used to center the data]
img = img.astype( numpy.float32 )

img[:,:,0] = (img[:,:,0] - ilsvrc_mean[0])
img[:,:,1] = (img[:,:,1] - ilsvrc_mean[1])
img[:,:,2] = (img[:,:,2] - ilsvrc_mean[2])

# Load the image as a half-precision floating point array
graph.LoadTensor( img.astype( numpy.float16 ), 'user object' )

# ---- Step 4: Read & print inference results from the NCS -------------------

# Get the results from NCS
output, userobj = graph.GetResult()

# Print the results
print('\n------- predictions --------')

labels = numpy.loadtxt( LABELS_FILE_PATH, str, delimiter = '\t' )

order = output.argsort()[::-1][:6]

for i in range( 0, 5 ):
	print ('prediction ' + str(i) + ' is ' + labels[order[i]])

skimage.io.imshow( IMAGES_PATH )
skimage.io.show( )

# ---- Step 5: Unload the graph and close the device -------------------------

graph.DeallocateGraph()
device.CloseDevice()

# ==== End of file ===========================================================

