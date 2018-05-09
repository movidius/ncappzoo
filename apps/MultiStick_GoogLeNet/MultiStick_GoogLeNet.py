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
images_folder = "../../data/images/"
labels_file = "../../data/ilsvrc12/synset_words.txt"
mean_file = "../../data/ilsvrc12/ilsvrc_2012_mean.npy"

if sys.version_info.major < 3 or sys.version_info.minor < 4:
    print("Please using python3.4 or greater!")
    exit(1)

if len(sys.argv) > 1:
    graph_folder = sys.argv[1]
    if len(sys.argv) > 2:
      images_folder = sys.argv[2]
else:
    print("WARNING: using", graph_folder, "for graph file")
    print("WARNING: using", images_folder, "for images dir")
    print("Run with python3 demo_ncs.py [graph_folder] [images_director] to change")
    input("Press enter to continue")

# ****************************************************************

from os import listdir, system, getpid
from mvnc import mvncapi as mvnc
import numpy
import cv2
from os.path import isfile, join
from subprocess import Popen as proc, PIPE
import random
from multiprocessing import Process, Queue as PQueue
from queue import Queue
from threading import Thread
import re
from tkinter import *

mvnc.SetGlobalOption(mvnc.GlobalOption.LOGLEVEL, 2)


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
dispQ = []

# Create n Queues based on the number of sticks plugged in
for devnum in range(len(devices)):
    dispQ.append(PQueue())

# *****************************************************************
# Read graph file, mean subtraction file and Categories
# *****************************************************************
#Load graph - this is the converted model from caffe
with open(graph_folder + "/graph", mode="rb") as f:
    graph = f.read()

#Load the Mean subtraction file from BVLC Caffe area
ilsvrc_mean = numpy.load(mean_file).mean(1).mean(1) #loading the mean file

#Load categories from ImageNet Labels
categories = []
categories = numpy.loadtxt(labels_file, str, delimiter="\t")

# *****************************************************************
# Read image file(s)
# *****************************************************************
imgarr = []
imgarr_orig = []
onlyfiles = [f for f in listdir(images_folder) if isfile(join(images_folder, f))]

# Only load the first 250 files, 
# so that we don"t exceed availible resources
onlyfiles = onlyfiles[:250]

print("Found ", len(onlyfiles), " images")
image_ext_list = [".jpg", ".png", ".JPEG", ".jpeg", ".PNG", ".JPG"]
# Preprocess images
for file in onlyfiles:
    fimg = images_folder + file
    if any([x in image_ext_list for x in fimg]):
        print(fimg + " is not an image file")
        continue
    img = cv2.imread(fimg)
    if img is None:
        print ("ERROR opening ", fimg)
        continue
    #print("Opened", fimg)
    
    img1 = cv2.resize(img, (700, 700))
    imgarr_orig.append(img1)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(numpy.float32)
    # Do mean substraction
    img[:,:,0] = (img[:,:,0] - ilsvrc_mean[0])
    img[:,:,1] = (img[:,:,1] - ilsvrc_mean[1])
    img[:,:,2] = (img[:,:,2] - ilsvrc_mean[2])
    img = img.astype(numpy.float16)
    imgarr.append(img)

print("Processed ", len(imgarr), " images")

# *****************************************************************
# Open the device and load the graph into each of the devices
# *****************************************************************
for devnum in range(len(devices)):
    print("***********************************************")
    devHandle.append(mvnc.Device(devices[devnum]))
    devHandle[devnum].OpenDevice()

    opt = devHandle[devnum].GetDeviceOption(mvnc.DeviceOption.OPTIMISATIONLIST)
    print("Optimisations:")
    print(opt)

    graphHandle.append(devHandle[devnum].AllocateGraph(graph))
    graphHandle[devnum].SetGraphOption(mvnc.GraphOption.ITERATIONS, 1)
    iterations = graphHandle[devnum].GetGraphOption(mvnc.GraphOption.ITERATIONS)
    print("Iterations:", iterations)

print("***********************************************")
print("Loaded Graphs")
print("***********************************************\n\n\n")

distTarget = 0
def runparallel(count=100, num=[], dispQ=0):
    #print("STARTED RUNPARALLEL")
    numdevices = num
    if len(num) == 0: numdevices = range(len(devices))
    
    while 1:
        imgchoice = []
        for devnum in numdevices:
            rnum = random.randint(0,len(imgarr)-1)
            imgchoice.append(rnum)
        # *****************************************************************
        # Load the Tensor to each of the devices
        # *****************************************************************
        for devnum in numdevices:
            #print("DEVICE", devnum, "loading data!")
            try:
                graphHandle[devnum].LoadTensor(imgarr[imgchoice[0]], "user object")
            except:
                pass
        # *****************************************************************
        # Read the result from each of the devices
        # *****************************************************************
        for devnum in numdevices:
            try:
                tensor, userobj = graphHandle[devnum].GetResult()
                order = tensor.argsort()
                last = len(order)-1
                predicted=int(order[last])
                #print("DEVICE", devnum, "putting data!")
                dispQ[devnum].put([imgchoice[0], categories[predicted]])
            except:
                print("ERROR ERROR ERROR!!!!")
                pass

# *****************************************************************
# Create and run threads to process images in parallel, 
# and start Display
# *****************************************************************
def startParallel(dispQ=dispQ, count=100, num=[]):

    print("Spawning processes!")  
 
    numdevices = num
    if len(num) == 0: numdevices = range(len(devices))
    thread_list = []
    statQ = PQueue()
    movP = Process(target=movProc, args=(count, dispQ, statQ, num))
    movP.start()
    # *****************************************************************
    # Make text Windows for both displays.
    # *****************************************************************
    root = Tk()
    T = Text(root, height=50, width=30)
    T.pack()
    root1 = Toplevel()
    if len(numdevices) > 1:
      T1 = Text(root1, height=50, width=30)
      T1.pack()
    root2 = Toplevel()
    T2 = Text(root2, height=50, width=50)
    T2.pack()

    q = PQueue()
    dtp = Process(target=displayThreadProcess, args=(q,))
    dtp.start()

    # *****************************************************************
    # Collect cpu stats to display
    # *****************************************************************    
    pid = statQ.get()
    dpid = str(getpid())
    print("got pid!", pid)
    statP = Process(target=statProc, args=((pid, dpid), statQ))
    statP.start()
    statT = Thread(target=statThread, args=(statQ, T2))
    statT.start()


    thread_list.append(Thread(target=displaythread, args=(dispQ, q, T, numdevices[:1],count,False)))
    if len(numdevices) > 1:
      thread_list.append(Thread(target=displaythread, args=(dispQ, q, T1, numdevices[1:],count,False)))

    for thread in thread_list:
        thread.start()
    root.mainloop()
    for thread in thread_list:
        thread.join()

def movProc(count, dispQ, statQ, num):
    numdevices = num
    pid = str(getpid())
    statQ.put(pid)
    if len(num) == 0: numdevices = range(len(devices))
    thread_list = []
    for ii in numdevices:
        thread_list.append(Thread(target=runparallel, args=(count,[ii],dispQ,)))
    for thread in thread_list:
        thread.start()
    for thread in thread_list:
        thread.join()

# Display stats
def statThread(Q, T):
    while 1:
        t = Q.get()
        if t == None:
            break
        t1 = Q.get()
        if t1 == None:
            break
        T.delete(1.0, END)
        T.insert(END, "Host CPU Utilization: "+t)
        T.insert(END, "\nDisp CPU Utilization: "+t1)
        T.see("end")

def statProc(pid, Q):

    # Get the fix point of a function f
    def fix(f, x):
        r = f(x)
        if x == r:
            return r
        return fix(f, r)

    # Collect stats using top from the processing running
    # inferencing and the display process
    while True:
        try:
            p = proc(["top", "-p", pid[0], "-b", "-n", "1"], stdout=PIPE)
            out, _ = p.communicate()
            lines = out.decode().split('\n')
            lines = list(filter(lambda x:len(x)>0, lines))
       
            s = lines[-1] 
            s = s.replace('\t', ' ')
            s = fix(lambda x:x.replace('  ', ' '), s)
            s = s.split(' ')
            if '%' in s[-4]:
              break
            #print(s[-4])
            Q.put(s[-4])

            p = proc(["top", "-p", pid[1], "-b", "-n", "1"], stdout=PIPE)
            out, _ = p.communicate()
            lines = out.decode().split('\n')
            lines = list(filter(lambda x:len(x)>0, lines))
       
            s = lines[-1] 
            s = s.replace('\t', ' ')
            s = fix(lambda x:x.replace('  ', ' '), s)
            s = s.split(' ')
            if '%' in s[-4]:
              break
            #print(s[-4])
            Q.put(s[-4])
        except:
            pass
    Q.put(None)
# *****************************************************************
# Thread that controls text window and 
# schedules frames for display windows
# *****************************************************************
def displaythread(dispQ, cvQ, T, num=[], count=100, block=True):
    numdevices = num
    if len(num) == 0: numdevices = range(len(devices))
    print("_--------------------------------_")
    print("numDevices:", numdevices)
    print("_--------------------------------_")

    while 1: 
        imgchoice = []
        for devnum in numdevices:
            img, label = (None, None)
            try:
              img, label = dispQ[devnum].get(block)
            except:
              continue        
 
            label = re.search("n[0-9]+\s([^,]+)", label).groups(1)[0]
            label2 = " ".join(["Stick",str(devnum), "->",label,"\n"])
            print("[DISPLAY]",label2)
            T.insert(END, label2)
            T.see("end")

            img = imgarr_orig[img]
            
            cvQ.put((img, label, len(numdevices)))
    T._root().destroy()

def displayThreadProcess(q):
    while 1:
        item = q.get()
        if item == None:
            break
        (img, label, n) = item
        label1 = "Running on " + str(n) + " Stick(s)"
        cv2.putText(img, label, (50,650), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 6)
        cv2.putText(img, label1, (25,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 6)
        cv2.imshow(label1, img)
        
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key is pressed, break from the loop
        if key == ord("q"):
            break
  
    # close any open windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    startParallel(count=500,num=[])

    # Clean up and close devices
    for devnum in range(len(devices)):
        graphHandle[devnum].DeallocateGraph()
        devHandle[devnum].CloseDevice()
    
    print("\n\nFinished\n\n")
