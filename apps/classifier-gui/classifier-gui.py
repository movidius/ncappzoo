#!/usr/bin/python3

# ****************************************************************************
# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.
# ****************************************************************************

# How to classify images using DNNs on Intel Neural Compute Stick (NCS)

import mvnc.mvncapi as mvnc
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

## Import the network library
sys.path.append('../../caffe/')
import GoogLeNet.run as GoogLeNet
import AlexNet.run as AlexNet
import SqueezeNet.run as SqueezeNet

# User modifiable input parameters
NCAPPZOO_PATH           = '../..'
IMAGE_PATH              = NCAPPZOO_PATH + '/data/images/cat.jpg'
LABELS_FILE_PATH        = NCAPPZOO_PATH + '/data/ilsvrc12/synset_words.txt'
IMAGE_MEAN              = [ 104.00698793, 116.66876762, 122.67891434]
IMAGE_STDDEV            = 1
IMAGE_DIM               = ( 224, 224 )

from os import system
system("sudo apt-get install -y python3-pil.imagetk")

root = Tk()
filename = StringVar()
filename.set(IMAGE_PATH)

NETWORKS = ["GoogLeNet", "AlexNet", "SqueezeNet"]

networkname = StringVar()
networkname.set(NETWORKS[0])

root.geometry('600x250+500+300')
root.title('Image Inference on Neural Compute Stick')

def quit():
        global root
        root.quit()

def buttonCallBack():
        root.filename = askopenfilename(filetypes = (("Image Files", "*.png"), ("All Files", "*.*")))
        filename.set(root.filename)

lblImage = None

def runInfer():
        network = networkname.get()
        IMAGE_PATH = filename.get()

        if network == "GoogLeNet":
            result,imgpath = GoogLeNet.infer(IMAGE_PATH)
        elif network == "AlexNet":
            result,imgpath = AlexNet.infer(IMAGE_PATH)
        elif network == "SqueezeNet":
            result,imgpath = SqueezeNet.infer(IMAGE_PATH)
        else:
            print("Network Not Supported Yet")
            messagebox.showinfo("Network Not Supported", "Network " + network + " is still WIP")
            return

        image = Image.open(imgpath)
        image = image.resize((100,100), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        lblImage = Label(image=photo, width=100, height=100)
        lblImage.image = photo
        lblImage.grid(row=2, column=0,pady=20)

        lblResult = Label(text=result)
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
qbtn.grid(row=3, column=3)

runInfer()

mainloop()





#def doSqueezeNet():
#        global lblImage
#        system("(cd ../../caffe/SqueezeNet/; test -f graph || make compile)")
#        GRAPH_PATH = NCAPPZOO_PATH + '/caffe/SqueezeNet/graph'
#        IMAGE_PATH = filename.get()
#        print(IMAGE_PATH)
#        if IMAGE_PATH:
#                devices = mvnc.EnumerateDevices()
#                if len( devices ) == 0:
#                        print( 'No devices found' )
#                        return
#                device = mvnc.Device( devices[0] )
#                device.OpenDevice()
#                with open( GRAPH_PATH, mode='rb' ) as f:
#                        blob = f.read()
#                graph = device.AllocateGraph( blob )
#                
#                img = skimage.io.imread( IMAGE_PATH )
#                img = skimage.transform.resize( img, IMAGE_DIM, preserve_range=True )
#                img = img[:, :, ::-1]
#                img = img.astype( numpy.float32 )
#                img = ( img - IMAGE_MEAN ) * IMAGE_STDDEV
#                graph.LoadTensor( img.astype( numpy.float16 ), 'user object' )
#                output, userobj = graph.GetResult()
#                print('\n------- predictions --------')
#                labels = numpy.loadtxt( LABELS_FILE_PATH, str, delimiter = '\t' )
#                order = output.argsort()[::-1][:6]
#                result = ""
#                for i in range( 0, 4 ):
#                    print ('prediction ' + str(i) + ' is ' + labels[order[i]])
#                    label = labels[order[i]]
#                    label = re.search("n[0-9]+\s([^,]+)", label).groups(1)[0]
#                    result = result + "\n%20s %0.2f %%" % (label, output[order[i]]*100)
#                graph.DeallocateGraph()
#                device.CloseDevice()
#                return result,IMAGE_PATH
#        else:
#                print("Image path is not set")
#                messagebox.showinfo("Image feild cannot be empty", "Choose the image you want to inference")
