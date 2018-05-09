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

dim=(224,224)
EXAMPLES_BASE_DIR='../../'
imagename = EXAMPLES_BASE_DIR+'data/images/nps_electric_guitar.png'

def infer(imgname):

        # get labels
        labels_file=EXAMPLES_BASE_DIR+'data/ilsvrc12/synset_words.txt'
        labels=numpy.loadtxt(labels_file,str,delimiter='\t')
        
        
        # Get a list of ALL the sticks that are plugged in
        devices = mvnc.enumerate_devices()
        if len(devices) == 0:
                print('No devices found')
                quit()
        
        # Pick the first stick to run the network
        device = mvnc.Device(devices[0])
        
        # Open the NCS
        device.open()

        # set the file name of the compiled network (graph file)
        file_dir = os.path.dirname(os.path.realpath(__file__))
        network_filename = file_dir + '/graph'

        # Load network graph file into memory
        with open(network_filename, mode='rb') as net_file:
                memory_graph = net_file.read()

        # create and allocate the graph object
        graph = mvnc.Graph("GoogLeNet Graph")
        fifo_in, fifo_out = graph.allocate_with_fifos(device, memory_graph)

        # Load the image and preprocess it for the network
        ilsvrc_mean = numpy.load(EXAMPLES_BASE_DIR+'data/ilsvrc12/ilsvrc_2012_mean.npy').mean(1).mean(1) #loading the mean file
        img = cv2.imread(imgname)
        img=cv2.resize(img,dim)
        img = img.astype(numpy.float32)
        img[:,:,0] = (img[:,:,0] - ilsvrc_mean[0])
        img[:,:,1] = (img[:,:,1] - ilsvrc_mean[1])
        img[:,:,2] = (img[:,:,2] - ilsvrc_mean[2])

        # Send the image to the NCS and queue an inference
        graph.queue_inference_with_fifo_elem(fifo_in, fifo_out, img.astype(numpy.float32), None)

        # Get the result from the NCS by reading the output fifo queue
        output, userobj = fifo_out.read_elem()
        
        # Print the results of the inference form the NCS
        order = output.argsort()[::-1][:6]
        print('\n------- predictions --------')
        result = ""
        for i in range(0,5):
                #print ('prediction ' + str(i) + ' (probability ' + str(output[order[i]]) + ') is ' + labels[order[i]] + '  label index is: ' + str(order[i]) )
                label = re.search("n[0-9]+\s([^,]+)", labels[order[i]]).groups(1)[0]
                result = result + "\n%20s %0.2f %%" % (label, output[order[i]]*100)
        
        # Clean up the graph, device, and fifos
        fifo_in.destroy()
        fifo_out.destroy()
        graph.destroy()
        device.close()
        device.destroy()

        return result, imgname

if __name__ == "__main__":
        result,imgname = infer(imagename)
        print (result)


