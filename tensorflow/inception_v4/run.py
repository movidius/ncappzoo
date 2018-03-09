#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.

from mvnc import mvncapi as mvnc
import sys
import numpy
import cv2

path_to_networks = './'
path_to_images = '../../data/images/'
graph_filename = 'graph'
image_filename = path_to_images + 'nps_electric_guitar.png'

#mvnc.SetGlobalOption(mvnc.GlobalOption.LOGLEVEL, 2)
devices = mvnc.EnumerateDevices()
if len(devices) == 0:
    print('No devices found')
    quit()

device = mvnc.Device(devices[0])
device.OpenDevice()

#Load graph
with open(path_to_networks + graph_filename, mode='rb') as f:
    graphfile = f.read()

#Load preprocessing data
mean = 128 
std = 1.0/128.0 

#Load categories
categories = []
with open(path_to_networks + 'categories.txt', 'r') as f:
    for line in f:
        cat = line.split('\n')[0]
        if cat != 'classes':
            categories.append(cat)
    f.close()
    print('Number of categories:', len(categories))

#Load image size
with open(path_to_networks + 'inputsize.txt', 'r') as f:
    reqsize = int(f.readline().split('\n')[0])

graph = device.AllocateGraph(graphfile)

img = cv2.imread(image_filename).astype(numpy.float32)

dx,dy,dz= img.shape
delta=float(abs(dy-dx))
if dx > dy: #crop the x dimension
    img=img[int(0.5*delta):dx-int(0.5*delta),0:dy]
else:
    img=img[0:dx,int(0.5*delta):dy-int(0.5*delta)]
    
img = cv2.resize(img, (reqsize, reqsize))

img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

for i in range(3):
    img[:,:,i] = (img[:,:,i] - mean) * std

print('Start download to NCS...')
graph.LoadTensor(img.astype(numpy.float16), 'user object')
output, userobj = graph.GetResult()

top_inds = output.argsort()[::-1][:5]

print(''.join(['*' for i in range(79)]))
print('inception-v4 on NCS')
print(''.join(['*' for i in range(79)]))
for i in range(5):
    print(top_inds[i], categories[top_inds[i]], output[top_inds[i]])

print(''.join(['*' for i in range(79)]))
graph.DeallocateGraph()
device.CloseDevice()
print('Finished')
