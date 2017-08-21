#! /usr/bin/env python3

from mvnc import mvncapi as mvnc
import sys
import numpy
import cv2
import time
import csv
import os
import sys

dim=(227,227)

# ***************************************************************
# get labels
# ***************************************************************
labels_file='synset_words.txt'
labels=numpy.loadtxt(labels_file,str,delimiter='\t')

# ***************************************************************
# configure the NCS
# ***************************************************************
mvnc.SetGlobalOption(mvnc.GlobalOption.LOGLEVEL, 2)

# ***************************************************************
# Get a list of ALL the sticks that are plugged in
# ***************************************************************
devices = mvnc.EnumerateDevices()
if len(devices) == 0:
	print('No devices found')
	quit()

# ***************************************************************
# Pick the first stick to run the network
# ***************************************************************
device = mvnc.Device(devices[0])

# ***************************************************************
# Open the NCS
# ***************************************************************
device.OpenDevice()
opt = device.GetDeviceOption(mvnc.DeviceOption.OPTIMISATIONLIST)

network_blob='graph'

#Load blob
with open(network_blob, mode='rb') as f:
	blob = f.read()

graph = device.AllocateGraph(blob)
graph.SetGraphOption(mvnc.GraphOption.ITERATIONS, 1)
iterations = graph.GetGraphOption(mvnc.GraphOption.ITERATIONS)

# ***************************************************************
# Load the image
# ***************************************************************
ilsvrc_mean = numpy.load('ilsvrc_2012_mean.npy').mean(1).mean(1) #loading the mean file
img = cv2.imread('images/cat.jpg')
img=cv2.resize(img,dim)
img = img.astype(numpy.float32)
img[:,:,0] = (img[:,:,0] - ilsvrc_mean[0])
img[:,:,1] = (img[:,:,1] - ilsvrc_mean[1])
img[:,:,2] = (img[:,:,2] - ilsvrc_mean[2])

# ***************************************************************
# Send the image to the NCS
# ***************************************************************
graph.LoadTensor(img.astype(numpy.float16), 'user object')

# ***************************************************************
# Get the result from the NCS
# ***************************************************************
output, userobj = graph.GetResult()

# ***************************************************************
# Print the results of the inference form the NCS
# ***************************************************************
order = output.argsort()[::-1][:6]
print('\n------- predictions --------')
for i in range(1,6):
	print ('prediction ' + str(i) + ' is ' + labels[order[i]])


# ***************************************************************
# Clean up the graph and the device
# ***************************************************************
graph.DeallocateGraph()
device.CloseDevice()
    



