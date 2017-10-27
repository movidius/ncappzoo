#! /usr/bin/env python3

# *******************************************************
# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.
# *******************************************************

# ****************************************************************************************
caffe_root = '/opt/movidius/caffe/'
imgroot = "../../data/images/"
# ****************************************************************************************

import sys
sys.path.insert(0, caffe_root + 'python')
import numpy as np
import os 
from os import *
from os.path import *
from multiprocessing import Process, Queue

onlyfiles = [f for f in os.listdir(imgroot) if isfile(join(imgroot, f))]
onlyfiles = onlyfiles[:100]

def caffeRun(dispQ):
  import caffe
  imgarr = []
  caffe.set_mode_cpu()
  system("(cd ../../caffe/GoogLeNet; make caffemodel; make prereqs)")
  model_def = caffe_root + 'models/bvlc_googlenet/deploy.prototxt'
  model_weights = '../../caffe/GoogLeNet/bvlc_googlenet.caffemodel'
  
  net = caffe.Net(model_def, model_weights, caffe.TEST)
  
  # load the mean ImageNet image (as distributed with Caffe) for subtraction
  mu = np.load('../../data/ilsvrc12/ilsvrc_2012_mean.npy')
  mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
  
  # create transformer for the input called 'data'
  transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
  
  transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
  transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
  transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
  transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
  
  print("Scanning files")
  image_ext_list = [".jpg", ".png", ".JPEG", ".jpeg", ".PNG", ".JPG"]
  for file in onlyfiles:
    fimg = imgroot + file
    if any([x in image_ext_list for x in fimg]):
        print(fimg + " is not an image file")
        continue
    img = caffe.io.load_image(fimg)
    transformed_image = transformer.preprocess('data', img)
    imgarr.append(transformed_image)
  print("Done! Starting computation")
  for i in range(len(imgarr)):
    dispQ.put((i, ""))
    img = imgarr[i]
    net.blobs['data'].data[...] = img
    output = net.forward()
    output_prob = output['prob'][0]
    print('predicted class is:', output_prob.argmax())
    labels_file = "../../data/ilsvrc12/synset_words.txt"
    labels = np.loadtxt(labels_file, str, delimiter='\t')
    output_label = labels[output_prob.argmax()]
    output_label = ' '.join(output_label.split(',')[0].split(' ')[1:])
    dispQ.put((i, output_label))
  dispQ.put((-1, None))

def cvPreprocess():
  import cv2
  imgarr_orig = []
  image_ext_list = [".jpg", ".png", ".JPEG", ".jpeg", ".PNG", ".JPG"]
  for file in onlyfiles:
    fimg = imgroot + file
    if any([x in image_ext_list for x in fimg]):
        print(fimg + " is not an image file")
        continue
    img1 = cv2.imread(fimg)
    if img1 is None:
        print ("ERROR opening ", fimg)
        continue
    img1 = cv2.resize(img1, (896, 896))
    imgarr_orig.append(img1)
  return imgarr_orig

def displaythread(dispQ):
  import cv2
  imgarr_orig = cvPreprocess()
  print("Got Display Images!")
  print("Starting Display Thread")
  last = (None, None)
  while 1:
    n = -1
    label = None
    try:
      (n, label) = dispQ.get(True, 1) 
      if n < 0:
        break
      last = (n, label)
    except:
      (n, label) = last
      if n == None:
        continue
    img = imgarr_orig[n]
    label = str(n)+" "+label
    cv2.putText(img, label, (100,850), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 6)
    # show the frame and record if the user presses a key
    cv2.imshow("Image", img)
    key = cv2.waitKey(1500) & 0xFF
    # if the `q` key is pressed, break from the loop
    if key == ord("q"):
       break

  # cleanup the camera and close any open windows
  cv2.destroyAllWindows()

if __name__=="__main__":
  dispQ = Queue()

  dthread = Process(target=displaythread, args=(dispQ,))
  cthread = Process(target=caffeRun, args=(dispQ,))

  cthread.start()
  dthread.start()

  cthread.join()
  dthread.join()

