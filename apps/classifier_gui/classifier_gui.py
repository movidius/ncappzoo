#!/usr/bin/python3

# ****************************************************************************
# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.
# ****************************************************************************

# How to classify images using DNNs on Intel Neural Compute Stick (NCS)

import skimage
from skimage import io, transform
import numpy
import os
import sys
import re
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk

GREEN = '\033[1;32m'
RED = '\033[1;31m'
NOCOLOR = '\033[0m'
YELLOW = '\033[1;33m'

try:
    from openvino.inference_engine import IENetwork, IEPlugin
except:
    print(RED + '\nPlease make sure your OpenVINO environment variables are set by sourcing the' + YELLOW + ' setupvars.sh ' + RED + 'script found in <your OpenVINO install location>/bin/ folder.\n' + NOCOLOR)
    exit(1)

## Import the network library
sys.path.append('../')
import simple_classifier_py.run as simple_classifier_py

GOOGLENET_IR =  '../../caffe/GoogLeNet/googlenet-v1.xml'
ALEXNET_IR =  '../../caffe/AlexNet/alexnet.xml'
SQUEEZENET_IR =  '../../caffe/SqueezeNet/squeezenet1.0.xml'

LABELS_FILE = '../../data/ilsvrc12/synset_labels.txt'

# User modifiable input parameters
NCAPPZOO_PATH           = '../..'
IMAGE_PATH              = NCAPPZOO_PATH + '/data/images/nps_mug.png'
LABELS_FILE_PATH        = NCAPPZOO_PATH + '/data/ilsvrc12/synset_labels.txt'

root = Tk()
filename = StringVar()
filename.set(IMAGE_PATH)

NETWORKS = ["GoogLeNet", "AlexNet", "SqueezeNet"]

networkname = StringVar()
networkname.set(NETWORKS[0])

root.geometry('600x250+500+300')
root.title('Image Inference on Intel Neural Compute Stick')

lblImage = None


def quit():
        global root
        root.quit()

def buttonCallBack():
        root.filename = askopenfilename(initialdir="../../data/images", filetypes = (("Image Files", "*.png *.jpg"), ("All Files", "*.*")))
        if len(root.filename) != 0:
            filename.set(root.filename)
            IMAGE_PATH = filename.get()
            image = Image.open(IMAGE_PATH)
            image = image.resize((100,100), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(image)
            lblImage = Label(image=photo, width=100, height=100)
            lblImage.image = photo
            lblImage.grid(row=2, column=0,pady=20)
            lblResult = Label(text="Press the 'What is this image?' button to continue.", anchor=W, justify=LEFT, height=10, width=100)
            lblResult.grid(row=2, column=1)


def runInfer():
		# Do the inference
        network = networkname.get()
        IMAGE_PATH = filename.get()
		
		# Update the window image and text
        image = Image.open(IMAGE_PATH)
        image = image.resize((100,100), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        lblImage = Label(image=photo, width=100, height=100)
        lblImage.image = photo
        lblImage.grid(row=2, column=0,pady=20)
        
        if network == "GoogLeNet":
            result, imgpath = simple_classifier_py.infer(IMAGE_PATH, GOOGLENET_IR, LABELS_FILE)
        elif network == "AlexNet":
            result, imgpath = simple_classifier_py.infer(IMAGE_PATH, ALEXNET_IR, LABELS_FILE)
        elif network == "SqueezeNet":
            result, imgpath = simple_classifier_py.infer(IMAGE_PATH, SQUEEZENET_IR, LABELS_FILE)
        else:
            print("Network Not Supported Yet")
            messagebox.showinfo("Network Not Supported", "Network " + network + " is still WIP")
            return

        
        lblResult = Label(text=result, anchor=W, justify=LEFT, height=10, width=100)
        lblResult.grid(row=2, column=1)
        print(result)
        
mEntry = Entry(root, textvariable=filename, width=45)
#mEntry.pack()
mEntry.grid(row=0, column=1, columnspan=2)

b = Button(root, text="Choose File", command=buttonCallBack)
#b.pack()
b.grid(row=0, column=0)

l = Label(root, text="Network Name")
#l.pack()
l.grid(row=1, column=0)

w=OptionMenu(root, networkname, *NETWORKS)
#w.pack()
w.grid(row=1, column=1, sticky=W)

btnInfer = Button(root, text="What is this image?", command=runInfer)
#btnInfer.pack()
btnInfer.grid(row=3, column=0)

qbtn = Button(root, text="Quit", command=quit)
#qbtn.pack()
qbtn.grid(row=3, column=1)

runInfer()

mainloop()


