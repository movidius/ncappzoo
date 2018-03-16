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

dim=(28,28)

imagename = '/home/neal/Downloads/mnist/testSample/img_1.jpg'

def infer(imgname):
        # ***************************************************************
        # get labels
        # ***************************************************************

        labels=[ 'digit 0', 'digit 1', 'digit 2', 'digit 3', 'digit 4', 'digit 5', 'digit 6', 'digit 7', 'digit 8', 'digit 9']
        

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
        network_blob = filefolder + '/mnist_inference.graph'

        #Load blob
        with open(network_blob, mode='rb') as f:
                blob = f.read()
        
        graph = device.AllocateGraph(blob)
        
        # ***************************************************************
        # Load the image
        # ***************************************************************
        mnist_mean = [128.0, 128.0, 128.0]
        img = cv2.imread(imgname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img=cv2.resize(img,dim)

        img = img.astype(numpy.float32)
        img[:] = ((img[:] - 128.0)*(1.0/255.0))
        #img[:,:,0] = ((img[:,:,0] - mnist_mean[0])*(1/255.0))
        #img[:,:,1] = ((img[:,:,1] - mnist_mean[1])*(1/255.0))
        #img[:,:,2] = ((img[:,:,2] - mnist_mean[2])*(1/255.0))
        
        # ***************************************************************
        # Send the image to the NCS
        # ***************************************************************
        graph.LoadTensor(img.astype(numpy.float16), 'user object')
        
        # ***************************************************************
        # Get the result from the NCS
        # ***************************************************************
        output, userobj = graph.GetResult()

        print ('-----------------------------------------------------------')
        print ('file: ' + imgname)
        print ('-----------------------------------------------------------')
        res_str = 'Raw inference results: ['
        for out_index in range(0, len(output) ):
            res_str = res_str + str(output[out_index])
            if (out_index != (len(output)-1)):
                res_str = res_str + ', '
        res_str = res_str + ']'
        print (res_str)

        print ('---')
        print ('Top 5 results from most ceretain to least:')
        print ('---')
        result = ''
        five_highest_indices = (-output).argsort()[:5]
        for index in range(0, 5):
            one_prediction = 'certainty ' + str(output[five_highest_indices[index]]) + ' --> ' + labels[five_highest_indices[index]]
            #print (one_prediction)
            result = result + one_prediction + '\n'


        # ***************************************************************
        # Print the results of the inference form the NCS
        # ***************************************************************
        #order = output.argsort()[::-1][:6]
        #print('\n------- predictions --------')
        #result = ""
        #for i in range(0,5):
        #        #print ('prediction ' + str(i) + ' (probability ' + str(output[order[i]]) + ') is ' + labels[order[i]] + '  label index is: ' + str(order[i]) )
        #        label = re.search("n[0-9]+\s([^,]+)", labels[order[i]]).groups(1)[0]
        #        result = result + "\n%20s %0.2f %%" % (label, output[order[i]]*100)
        
        # ***************************************************************
        # Clean up the graph and the device
        # ***************************************************************
        graph.DeallocateGraph()
        device.CloseDevice()

        return result, imgname

if __name__ == "__main__":
        result,imgname = infer(imagename)
        print (result)
        print ('-----------------------------------------------------------')



