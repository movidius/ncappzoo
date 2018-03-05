#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#~ The MIT License (MIT)
#~ Copyright 2018 ©klo86min
#~ Permission is hereby granted, free of charge, to any person obtaining a copy 
#~ of this software and associated documentation files (the "Software"), to deal 
#~ in the Software without restriction, including without limitation the rights 
#~ to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
#~ copies of the Software, and to permit persons to whom the Software is 
#~ furnished to do so, subject to the following conditions:
#~ The above copyright notice and this permission notice shall be included in 
#~ all copies or substantial portions of the Software.
#~ THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
#~ IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
#~ FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
#~ AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
#~ LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
#~ OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
#~ SOFTWARE.

import argparse
import csv
import cv2
import mvnc.mvncapi as mvnc
import numpy as np
import os.path

# paths settings (see also data argument parameter)
INFERENCE_FILE = "../inferences.csv"
GRAPH_FILE = "../compiled.graph"
EXAMPLE_ONLY = False
if EXAMPLE_ONLY:
    DATA_DIR = "training"
    IMAGES_FILE = "training_ground_truth.csv"
    LAST_IMAGE = 2000
else:    
    DATA_DIR = "provisional"
    IMAGES_FILE = "provisional.csv"
    LAST_IMAGE = None

# image settings
IMAGE_MEAN = (108, 118, 128)
IMAGE_SCALE = 1.0 / 58.0
IMAGE_DIM = (256, 256)

###############################################################################
#
# Modified code from https://github.com/ashwinvijayakumar/ncappzoo/apps/
# rapid-image-classifier/rapid-image-classifier.py
# also under the MIT License
#
###############################################################################

# ---- Step 1: Open the enumerated device and get a handle to it -------------

def open_ncs_device(verbose=False):
    if verbose:
        mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)
    # Look for enumerated NCS device(s); quit program if none found.
    devices = mvnc.EnumerateDevices()
    if len( devices ) == 0:
        print( 'No devices found' )
        quit()

    # Get a handle to the first enumerated device and open it
    device = mvnc.Device( devices[0] )
    device.OpenDevice()

    return device

# ---- Step 2: Load a graph file onto the NCS device -------------------------

def load_graph( device, graph_file):

    # Read the graph file into a buffer
    with open( graph_file, mode='rb' ) as f:
        blob = f.read()

    # Load the graph buffer into the NCS
    graph = device.AllocateGraph( blob )

    return graph

# ---- Step 5: Unload the graph and close the device -------------------------

def close_ncs_device( device, graph ):
    graph.DeallocateGraph()
    device.CloseDevice()

#####################   End of ncappzoo code   ################################

class MovidiusImage(object):
    """Image metadata and loader for Movidius NCS
    
    Args:
        name (str): image reference name as used in CSV files
        path (str): image path
        class_index (int): 1-based class label index
    
    Attributes:
        top_k (list): list of predicted (class_index, proba)
        inference_time (float): computation time in ms 
    """
    
    def __init__(self, name, path, class_index = None):
        self.name = name
        self.path = path
        self.class_index = class_index
        self.top_k = None
        self.inference_time = None
        
    def load_BGR(self, dim, mean, scale, dtype=np.float16):
        """Return image data in BGR order
        
        Args:
            dim (tuple): image dimensions
            mean (tuple): mean values to be substracted
            scale (float): scaling factor (that is 1.0 / std)
            dtype (numpy.dtype): new type for the BGR blob
        
        Returns:
            numpy.ndarray: the transformed BGR blob
        """
        # Both OpenCV and Caffe use BGR as default 
        img = cv2.imread(self.path)
        img = cv2.resize(img, dim)
        # Temporary switch to np.float32 to apply transformations
        img = img.astype(np.float32)
        for channel in range(3):
            img[:,:,channel] = (img[:,:,channel] - mean[channel]) * scale
        img = img.astype(dtype)
        return img
        
    def save_top_k(self, predictions, k=5):
        """Save the top_k predicted probabilities
        
        Args:
            predictions (numpy.ndarray): the probabilities for each class
            k (int): Number of top_k probas
        """
        order_k = predictions.argsort()[::-1][:k]
        # class_index is 1-based
        self.top_k = [(pos + 1, np.float(predictions[pos])) 
            for pos in order_k]

    
    def result_string(self):
        """ Return image results with the following fields:
        [name, top1, proba1, ... top5, proba5, time]
        
        Returns:
            str: formatted CSV string
        """
        res = [ self.name, ]
        for k, prob in self.top_k:
            res += [k, prob]
        res += [self.inference_time]
        pattern = "%s," + "%d,%.9f," * len(self.top_k) + "%.9f"
        return pattern % tuple(res)
        
def init_images(data_dir, images_file):
    """Parse image_file CSV and create one MovidiusImage per row.
    
    Args:
        data_dir (str): path of the folder containing images
        image_file (str): CSV file (one image path per row)
    
    Returns:
        list: list of MovidiusImage instances
    """
    images = []
    with open(images_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # skip header
        next(reader)
        for row_pos, row in enumerate(reader):
            name = row[0]
            path = os.path.join(data_dir, name)
            class_index = int(row[1]) if len(row) > 1 else None
            images.append(MovidiusImage(name, path, class_index))
    return images
            
def write_inferences_csv(output_path, images):
    """ For each image, retrieve and write results.
    
    Args:
        output_path (str): path for the CSV output
        images (list): list of processed MovidiusImage instances
    """
    with open(output_path, 'w') as output_file:
        for image in images:
            output_file.write(image.result_string() + '\n')

def score_inferences(images, min_proba = 1e-15, mult = 100, n_classes=200, 
    log_loss_max=15, time_limit=1000.0):
    """ Compute the logLoss and reference computation time
    
    Args:
        images (list): list of processed MovidiusImage instances
        min_proba (float): minimum probability to be used in logLoss
        mult (int): number of images used for the reference time
        n_classes (int): total number of classes
        log_loss_limit (float): minimum log_loss requirement
        time_limit (float): maximum time per image (in ms)
        
    Returns:
        tuple: LogLoss and reference_time float values
    """
    min_proba = np.float(min_proba)
    max_proba = 1.0 - min_proba
    n_images = len(images)
    probas = np.zeros(n_images, dtype=np.float)
    image_time = 0.0
    top_1_accuracy = 0.0
    top_k_accuracy = 0.0
    for i, image in enumerate(images):
        class_probas = dict(image.top_k)
        if image.class_index == image.top_k[0][0]:
            top_1_accuracy += 1.0            
        if image.class_index in class_probas:
            top_k_accuracy += 1.0
            probas[i] = class_probas[image.class_index]
        if probas[i] > 0:
            sum_probas = sum(class_probas.values())
            probas[i] /= sum_probas
        probas[i] = max(min_proba, min(max_proba, probas[i]))
        image_time += image.inference_time
   
    log_loss = np.mean(-np.log(probas))
    top_1_accuracy /=  n_images
    top_k_accuracy /=  n_images
    image_time /= n_images
    t = mult * image_time
    print("top_1_accuracy = %.9f" % top_1_accuracy)
    print("top_k_accuracy = %.9f" % top_k_accuracy )
    print("log_loss = %.9f" % log_loss)
    print("image_time = %.9f" % image_time)
    if image_time > time_limit or log_loss > log_loss_max:
        score = 0.0
    else:
        t_max = mult * time_limit
        score = 1e6 * (1.0 - log_loss * np.log(t) / (log_loss_max *  np.log(t_max)))
    print("score = %.2f" % score)
    return score
    

def main(args):
    parser = argparse.ArgumentParser(description='TopCoder Movidius MM')
    parser.add_argument(
        "data",
        help="""Data directory for Movidius NCS Challenge (if necessary, see also DATADIR variable and 'data' target in Makefile), expected folder setup is:
        data/
        data/provisional/
        data/provisional/provisional_{00001..02000}.jpg
        data/training/
        data/training/training_{00001..80000}.jpg
        data/training_ground_truth.csv
        data/provisional.csv"""
    )
    args = parser.parse_args()
    if not os.path.isdir(args.data):
        print("data is not a directory: %s" % args.data)
        print("Please use the right path as argument, and/or change the Makefile MOVIDIUSDIR variable")
        return 0
    
    print("IMAGE_MEAN", IMAGE_MEAN)
    print("IMAGE_SCALE", IMAGE_SCALE)
    print("IMAGE_DIM", IMAGE_DIM)
    # start NCS
    device = open_ncs_device()
    graph = load_graph(device, GRAPH_FILE)
    # prepare images
    data_dir = os.path.join(args.data, DATA_DIR)
    images_file = os.path.join(args.data, IMAGES_FILE)
    images = init_images(data_dir, images_file)
    if LAST_IMAGE is not None:
        images = images[:LAST_IMAGE]
    n_images = len(images)
    info_frequency = 100
    print("n_images = %d" % n_images)
    # process images
    for i, image in enumerate(images):
        if (i+1) % info_frequency == 0:
            print("progess %d/%d ..." % (i+1, n_images), flush=True)
        bgr_blob = image.load_BGR(IMAGE_DIM, IMAGE_MEAN, IMAGE_SCALE)
        graph.LoadTensor(bgr_blob, 'user object')
        output, userobj = graph.GetResult()
        image.inference_time = np.sum(
            graph.GetGraphOption( mvnc.GraphOption.TIME_TAKEN ) )
        image.save_top_k(output, 5)
    # stop NCS
    close_ncs_device(device, graph)
    # process results
    write_inferences_csv(INFERENCE_FILE, images)
    if EXAMPLE_ONLY:
        score_inferences(images)
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
