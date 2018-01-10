#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.

from mvnc import mvncapi as mvnc
import sys
import numpy
import cv2
import time
import csv
import os
import sys
import re
from os import system

dim=(227,227)
EXAMPLES_BASE_DIR='../../'
imagename = EXAMPLES_BASE_DIR+'data/images/nps_electric_guitar.png'

def infer(imgname):
        # ***************************************************************
        # get labels
        # ***************************************************************
        labels_file=EXAMPLES_BASE_DIR+'data/ilsvrc12/synset_words.txt'
        labels=numpy.loadtxt(labels_file,str,delimiter='\t')
        
        # ***************************************************************
        # configure the NCS
        # ***************************************************************
        mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)
        
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
        
        filefolder = os.path.dirname(os.path.realpath(__file__)) 
        network_blob = filefolder + '/graph'
        system('(cd ' + filefolder + ';test -f graph || make compile)')
        
        #Load blob
        with open(network_blob, mode='rb') as f:
                blob = f.read()
        
        graph = device.AllocateGraph(blob)
        
        # ***************************************************************
        # Load the image
        # ***************************************************************
        ilsvrc_mean = numpy.load(EXAMPLES_BASE_DIR+'data/ilsvrc12/ilsvrc_2012_mean.npy').mean(1).mean(1) #loading the mean file
        img = cv2.imread(imgname)
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
        result = ""
        for i in range(0,5):
                #print ('prediction ' + str(i) + ' (probability ' + str(output[order[i]]*100) + '%) is ' + labels[order[i]] + '  label index is: ' + str(order[i]) )
                label = re.search("n[0-9]+\s([^,]+)", labels[order[i]]).groups(1)[0]
                result = result + "\n%20s %0.2f %%" % (label, output[order[i]]*100)

        # ***************************************************************
        # Clean up the graph and the device
        # ***************************************************************
        graph.DeallocateGraph()
        device.CloseDevice()
        
        return result, imgname
    
if __name__ == "__main__":
        result,imgname = infer(imagename)
        print (result)
