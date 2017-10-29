#! /usr/bin/env python3

# *******************************************************
# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.
# *******************************************************


# *******************************************************
# Demo with GoogleNet V1 and trained with ImageNet database
# *******************************************************

import sys
graph_folder="../../caffe/GoogLeNet/"
images_folder="../../data/images/"
if sys.version_info.major < 3 or sys.version_info.minor < 4:
  print("Please using python3.4 or greater!")
  exit(1)

if len(sys.argv) > 1:
  graph_folder = sys.argv[1]
  if len(sys.argv) > 2:
    images_folder = sys.argv[2]
else:
  print("WARNING: using", graph_folder, "for graph file")
  print("Run with python3 MultiStick_GoogLeNet_Camera.py [graph_folder] to change")
  input("Press enter to continue")

# ****************************************************************

from mvnc import mvncapi as mvnc
import numpy
import cv2
from os import system
from os.path import isfile, join
from queue import Queue
from threading import Thread, Event, Lock
import re
from time import sleep

mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)

# *****************************************************************
# Get a list of devices
# *****************************************************************
devices = mvnc.EnumerateDevices()
if len(devices) == 0:
  print("No devices found")
  quit()
print(devices)
devHandle   = []
graphHandle = []


# *****************************************************************
# Read graph file, mean subtraction file and Categories
# *****************************************************************
#Load graph - this is the converted model from caffe
with open(join(graph_folder, "graph"), mode="rb") as f:
  graph = f.read()

#Load the Mean subtraction file from BVLC Caffe area
ilsvrc_mean = numpy.load("../../data/ilsvrc12/ilsvrc_2012_mean.npy").mean(1).mean(1) #loading the mean file

#Load categories from ImageNet Labels
categories = []
labels_file = "../../data/ilsvrc12/synset_words.txt"
categories = numpy.loadtxt(labels_file, str, delimiter="\t")

# *****************************************************************
# Open the device and load the graph into each of the devices
# *****************************************************************
for devnum in range(len(devices)):
  print("***********************************************")
  devHandle.append(mvnc.Device(devices[devnum]))
  devHandle[devnum].OpenDevice()

  opt = devHandle[devnum].GetDeviceOption(mvnc.DeviceOption.OPTIMISATION_LIST)
  print("Optimisations:")
  print(opt)

  graphHandle.append(devHandle[devnum].AllocateGraph(graph))
  graphHandle[devnum].SetGraphOption(mvnc.GraphOption.ITERATIONS, 1)
  iterations = graphHandle[devnum].GetGraphOption(mvnc.GraphOption.ITERATIONS)
  print("Iterations:", iterations)

print("***********************************************")
print("Loaded Graphs")
print("***********************************************\n\n\n")


# *****************************************************************
# Set up variables for communicating with camera 
# *****************************************************************
cam = cv2.VideoCapture(0)
lock = Lock()
frameBuffer = []
results = Queue()

# *****************************************************************
# Thread for displaying camera images without lag 
# *****************************************************************
def camThread(cam, lock, buff, resQ):
  lastlabel = ""
  print("press 'q' to quit!")
  while 1:
    s, img = cam.read()
    if not s:
      print("Could not get frame")
      continue
    lock.acquire()
    if len(buff)>10:
      for i in range(10):
        del buff[0]
    buff.append(img)
    lock.release()
    label = None
    try:
      label = resQ.get(False)
    except:
      pass

    img1 = cv2.resize(img, (700, 700))
    # *****************************************************************
    # Get label if one exists 
    # *****************************************************************
    if label == None:
      cv2.putText(img1, lastlabel, (50,650), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 6)
    else:
      cv2.putText(img1, label, (50,650), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 6)
      lastlabel = label
    cv2.imshow('Camera', img1)
    
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the loop
    if key == ord("q"):
        break
 
  # close any open windows
  cv2.destroyAllWindows()
  
  lock.acquire()
  while len(buff) > 0:
    del buff[0]
  lock.release()

def inferencer(results, lock, frameBuffer, handle):
  failure = 0
  # Sleep to let camera initialize
  sleep(1)
  while failure < 100:
    # Check if the buffer has a frame
    # If it doesn't, record it as a failure
    # Exits after 100 continuous failures
    lock.acquire()
    if len(frameBuffer) == 0:
      lock.release()
      failure += 1
      continue
    img = frameBuffer[-1].copy()
    del frameBuffer[-1]
    failure = 0
    lock.release()
  
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(numpy.float32)
    # Do mean substraction
    img[:,:,0] = (img[:,:,0] - ilsvrc_mean[0])
    img[:,:,1] = (img[:,:,1] - ilsvrc_mean[1])
    img[:,:,2] = (img[:,:,2] - ilsvrc_mean[2])
    img = img.astype(numpy.float16)
  
    # Compute image label on NCS
    handle.LoadTensor(img, "user object")
    tensor, userobj = handle.GetResult()

    order = tensor.argsort()
    last = len(order)-1
    predicted=int(order[last])
    label = categories[predicted]
  
    label = re.search("n[0-9]+\s([^,]+)", label).groups(1)[0]
   
    # Send label back to camera thread so it can be displayed
    results.put(label)

# Start all threads
threads = []

camT = Thread(target=camThread, args=(cam, lock, frameBuffer, results))
camT.start()
threads.append(camT)

for devnum in range(len(devices)):
  t = Thread(target=inferencer, args=(results, lock, frameBuffer, graphHandle[devnum]))
  t.start()
  threads.append(t)

# Wait for all threads to finish
for t in threads:
  t.join()

# Clean up and close devices
for devnum in range(len(devices)):
  graphHandle[devnum].DeallocateGraph()
  devHandle[devnum].CloseDevice()

print("\n\nFinished\n\n")
