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

IMGCLASS = "1.0"
IMGSIZE = "224"

#Load preprocessing data
mean = 127.5 
std = 1/127.5

if len(sys.argv) > 1:
    IMGCLASS = sys.argv[1]
    if len(sys.argv) > 2:
      IMGSIZE = sys.argv[2]
else:
    print("WARNING: using", IMGCLASS, " for MobileNet Class")
    print("WARNING: using", IMGSIZE, " for MobileNet image size")
    print("Run with ./run.py [IMGCLASS] [IMGSIZE] to change")


reqsize = int(IMGSIZE)

# ***************************************************************
# Preprocessor Routines
# ***************************************************************
def center_crop(img):
    print("Performing Center Crop")
    dx,dy,dz= img.shape
    delta=float(abs(dy-dx))
    if dx > dy: #crop the x dimension
        img=img[int(0.5*delta):dx-int(0.5*delta),0:dy]
    else:
        img=img[0:dx,int(0.5*delta):dy-int(0.5*delta)]
    return img

def load_preprocess_image(fimg, dim, mean, scale, colorsequence="BGR", centercrop=False):
        # Load the image using open cv which reads the image as BGR
        img_orig = cv2.imread(fimg)
        if img_orig is None:
            print ("ERROR opening ", fimg)
            return None, None

        img = img_orig.astype(numpy.float32)
        
        if centercrop: # Center Crop
            img = center_crop(img)

        print("Resizing to ", dim)
        img = cv2.resize(img, dim) # Resize Image
        if colorsequence == "RGB": # If the colorsequence is RGB, color convert
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            print("Converting image to RGB")

        print("Mean subtracting and scaling image Mean=", mean, " Scale=", scale)
        img = img - mean           # Subtract Mean
        img = img * scale          # Scale the Image
        return (img_orig, img.astype(numpy.float16))

devices = mvnc.EnumerateDevices()
if len(devices) == 0:
    print('No devices found')
    quit()

device = mvnc.Device(devices[0])
device.OpenDevice()

#Load graph
with open(path_to_networks + graph_filename, mode='rb') as f:
    graphfile = f.read()


#Load categories
categories = []
with open(path_to_networks + 'categories.txt', 'r') as f:
    for line in f:
        cat = line.split('\n')[0]
        if cat != 'classes':
            categories.append(cat)
    f.close()
    print('Number of categories:', len(categories))

#  #Load image size
#  with open(path_to_networks + 'inputsize.txt', 'r') as f:
#      reqsize = int(f.readline().split('\n')[0])

graph = device.AllocateGraph(graphfile)

img1, img = load_preprocess_image(image_filename, (reqsize, reqsize), mean, std, "RGB", True)
#print(img)
print('Start download to NCS...')
graph.LoadTensor(img, 'user object')
output, userobj = graph.GetResult()
#print(output)

top_inds = output.argsort()[::-1][:5]

print(''.join(['*' for i in range(79)]))
print('Mobilenet on NCS')
print(''.join(['*' for i in range(79)]))
for i in range(5):
    print(top_inds[i], categories[top_inds[i]], output[top_inds[i]])

print(''.join(['*' for i in range(79)]))
graph.DeallocateGraph()
device.CloseDevice()
print('Finished')
