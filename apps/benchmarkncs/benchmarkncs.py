#! /usr/bin/env python3
# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.

from mvnc import mvncapi as mvnc
from sys import argv
import numpy
import cv2
from os import listdir
from os.path import isfile, join
from random import choice
from timeit import timeit
from threading import Thread
from os import system

if len(argv) != 5:
	print('Syntax: python3 pytest.py <network directory> <picture directory> <input img width> <input img height>')
	print('        <network directory> is the directory that contains graph, stat.txt and')
	print('                            categories.txt')
	print('        <picture directory> is the directory with several JPEG or PNG images to process')
	quit()
	
img_width  = int(argv[3])
img_height = int(argv[4])

#mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)

# *****************************************************************
# Get a list of devices
# *****************************************************************
devices = mvnc.EnumerateDevices()
if len(devices) == 0:
	print('No devices found')
	quit()
#print("\n\nFound ", len(devices), "devices :", devices, "\n\n")
devHandle   = []
graphHandle = []

# *****************************************************************
# Read and preprocess image file(s)
# *****************************************************************
imgarr = []
onlyfiles = [f for f in listdir(argv[2]) if isfile(join(argv[2], f))]
onlyfiles = onlyfiles[:100]

for file in onlyfiles:
    fimg = argv[2] + "/" + file
    print("Opening file ", fimg)
    img = cv2.imread(fimg)
    #if img==None:
    #  continue
    img = cv2.resize(img, (img_width, img_height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(numpy.float16)
    #print(file, img.shape)
    imgarr.append(img)

# *****************************************************************
# Read graph file
# *****************************************************************
graph_folder=argv[1]
system("(cd " + graph_folder + "; test -f graph || make compile)")

#Load graph
with open(join(argv[1],'graph'), mode='rb') as f:
	graph = f.read()

# *****************************************************************
# Open the device and load the graph into each of the devices
# *****************************************************************
for devnum in range(len(devices)):
    #print("***********************************************")
    
    devHandle.append(mvnc.Device(devices[devnum]))
    devHandle[devnum].OpenDevice()
            
    opt = devHandle[devnum].GetDeviceOption(mvnc.DeviceOption.OPTIMISATION_LIST)
    #print("Optimisations:")
    #print(opt)

    graphHandle.append(devHandle[devnum].AllocateGraph(graph))
    graphHandle[devnum].SetGraphOption(mvnc.GraphOption.ITERATIONS, 1)
    iterations = graphHandle[devnum].GetGraphOption(mvnc.GraphOption.ITERATIONS)
    #print('Iterations:', iterations)

#print("***********************************************")
#print("Loaded Graphs")
#print("***********************************************\n\n\n")

def runparallel(count=100, num=[]):
    numdevices = num
    if len(num) == 0: numdevices = range(len(devices))

    for i in range(count):
        # *****************************************************************
        # Load the Tensor to each of the devices
        # *****************************************************************
        for devnum in numdevices:
            img = choice(imgarr)
            graphHandle[devnum].LoadTensor(img, 'user object')
    
        # *****************************************************************
        # Read the result from each of the devices
        # *****************************************************************
        for devnum in numdevices:
            tensor, userobj = graphHandle[devnum].GetResult()

def runthreaded(count=100,num=[]):
    numdevices = num
    if len(num) == 0: numdevices = range(len(devices))
    thread_list = []
    for ii in numdevices:
        thread_list.append(Thread(target=runparallel, args=(count,[ii],)))

    for thread in thread_list:
        thread.start()
    for thread in thread_list:
        thread.join()

if __name__ == '__main__':
    # *****************************************************************
    # Runs and times runthreaded with 'i' sticks, until all sticks 
    # run at once
    # *****************************************************************
    for i in range(1, len(devices)+1):
      num = str(list(range(i))) 
      tot_time = timeit("runthreaded(count=100,num="+num+")", setup="from __main__ import runthreaded", number=1)    
      print("\n\nRunning " + argv[1] +" on "+str(i)+" sticks threaded      : %0.2f FPS\n\n"%(100.0*i/tot_time))


# *****************************************************************
# Clean up and close devices 
# *****************************************************************
for devnum in range(len(devices)):
    graphHandle[devnum].DeallocateGraph()
    devHandle[devnum].CloseDevice()

#print('\n\nFinished\n\n')
