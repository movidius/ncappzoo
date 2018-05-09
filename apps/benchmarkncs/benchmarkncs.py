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

mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, 2)

# *****************************************************************
# Get a list of devices
# *****************************************************************

devices = mvnc.enumerate_devices()
if len(devices) == 0:
	print('No devices found')
	quit()
print("num devices: ", len(devices))
dev_handle = []
graph_handle = []

# *****************************************************************
# Open the NCS device
# *****************************************************************

graph_folder=argv[1]
system("(cd " + graph_folder + "; test -f graph || make compile)")

graph_file_path = join(argv[1],'graph')

# *****************************************************************
# Read and preprocess image file(s)
# *****************************************************************
imgarr = []
selected_files = [f for f in listdir(argv[2]) if isfile(join(argv[2], f))]
selected_files = selected_files[:100]

for file in selected_files:
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
# Open the device, create fifos and load the graph into each of the devices
# *****************************************************************
fifoIn_handle = [] 
fifoOut_handle = [] 

descIn = [] 
descOut = [] 

for dev_num in range(len(devices)):
    # add a ncs device to the device handle
    dev_handle.append(mvnc.Device(devices[dev_num]))
    # open the device
    dev_handle[dev_num].open()
    # load blob
    with open(graph_file_path, mode='rb') as network_file:
        blob = network_file.read()
    # initialize the graph then allocate the graph on the current device
    graph_handle.append(mvnc.Graph(graph_file_path))

    # set up the input and output queues for the current device
    fifoIn, fifoOut = graph_handle[dev_num].allocate_with_fifos(dev_handle[dev_num], blob)
    fifoIn_handle.append(fifoIn)
    fifoOut_handle.append(fifoOut)
    
#print("***********************************************")
#print("Loaded Graphs")
#print("***********************************************\n\n\n")

def runparallel(count=100, num=[]):
    num_devices = num
    if len(num) == 0: num_devices = range(len(devices))

    for i in range(count):
        # *****************************************************************
        # Load the Tensor to each of the devices
        # *****************************************************************
        for dev_num in num_devices:
            img = choice(imgarr)
            # send the 
            graph_handle[dev_num].queue_inference_with_fifo_elem(fifoIn_handle[dev_num], fifoOut_handle[dev_num], img.astype(numpy.float32), None)
        # *****************************************************************
        # Read the result from each of the devices
        # *****************************************************************
        for dev_num in num_devices:
            tensor, userobj = fifoOut_handle[dev_num].read_elem()

def runthreaded(count=100,num=[]):
    num_devices = num
    if len(num) == 0: num_devices = range(len(devices))
    thread_list = []
    for ii in num_devices:
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
# Close/clean up fifos, graphs, and devices
# *****************************************************************
for f_handle in fifoIn_handle + fifoOut_handle:
    f_handle.destroy()
for g_handle in graph_handle:
    g_handle.destroy()
for d_handle in dev_handle:
    d_handle.close()
    d_handle.destroy()

#print('\n\nFinished\n\n')
