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


def do_initialize():
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

        # Load blob
        with open(network_blob, mode='rb') as f:
                blob = f.read()

        graph = device.AllocateGraph(blob)

        return device, graph


def do_inference(graph, imgname):
        # ***************************************************************
        # get labels
        # ***************************************************************

        labels=[ 'digit 0', 'digit 1', 'digit 2', 'digit 3', 'digit 4', 'digit 5', 'digit 6', 'digit 7', 'digit 8', 'digit 9']
        

        # ***************************************************************
        # Load the image
        # ***************************************************************
        mnist_mean = 128.0
        img = cv2.imread(imgname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img=cv2.resize(img,dim)
        img = img.astype(numpy.float32)

        #img[:] = ((img[:]) )
        img[:] = ((img[:] )*(1.0/255.0))

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

        #printing the raw results,
        #res_str = 'Raw inference results: ['
        #for out_index in range(0, len(output) ):
        #    res_str = res_str + str(output[out_index])
        #    if (out_index != (len(output)-1)):
        #        res_str = res_str + ', '
        #res_str = res_str + ']'
        #print (res_str)

        # sort indices in order of highest probabilities
        five_highest_indices = (-output).argsort()[:5]

        inference_labels = []
        inference_probabilities = []

        for index in range(0, 5):
            inference_probabilities.append(str(output[five_highest_indices[index]]))
            inference_labels.append(labels[five_highest_indices[index]])

        return inference_labels, inference_probabilities

def do_cleanup(device, graph):
        graph.DeallocateGraph()
        device.CloseDevice()

def show_inference_results(imagename, infer_labels, infer_probabilities):
        print('-----------------------------------------------------------')
        print('file: ' + imagename + '--> ' + infer_labels[0])
        print('-----------------------------------------------------------')

        print('---')
        print('Top 5 results from most certain to least:')
        print('---')

        for index in range(0, 5):
                one_prediction = 'certainty ' + str(infer_probabilities[index]) + ' --> ' + infer_labels[index]
                print(one_prediction)

        print('-----------------------------------------------------------')


if __name__ == "__main__":
        image_name_list = []
        image_name_list.append('./digit_images/zero.png')
        image_name_list.append('./digit_images/one.png')
        image_name_list.append('./digit_images/two.png')
        image_name_list.append('./digit_images/three.png')
        image_name_list.append('./digit_images/four.png')
        image_name_list.append('./digit_images/five.png')
        image_name_list.append('./digit_images/six.png')
        image_name_list.append('./digit_images/seven.png')
        image_name_list.append('./digit_images/eight.png')
        image_name_list.append('./digit_images/nine.png')

        device, graph = do_initialize()

        for index in range(0, len(image_name_list)):
            infer_labels, infer_probabilities = do_inference(graph, image_name_list[index])
            show_inference_results(image_name_list[index], infer_labels, infer_probabilities)

        do_cleanup(device, graph)




