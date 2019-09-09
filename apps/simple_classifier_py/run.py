#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory. 

import sys
import numpy
import cv2
import argparse

GREEN = '\033[1;32m'
RED = '\033[1;31m'
NOCOLOR = '\033[0m'
YELLOW = '\033[1;33m'
DEVICE = "MYRIAD"

try:
    from openvino.inference_engine import IENetwork, IECore
except:
    print(RED + '\nPlease make sure your OpenVINO environment variables are set by sourcing the' + YELLOW + ' setupvars.sh ' + RED + 'script found in <your OpenVINO install location>/bin/ folder.\n' + NOCOLOR)
    exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description = 'Image classifier using \
                         Intel® Neural Compute Stick 2.' )
    parser.add_argument( '--ir', metavar = 'IR_File',
                        type=str, default = '../../caffe/SqueezeNet/squeezenet_v1.0.xml', 
                        help = 'Absolute path to the neural network IR file.')
    parser.add_argument( '-l', '--labels', metavar = 'LABEL_FILE', 
                        type=str, default = '../../data/ilsvrc12/synset_labels.txt',
                        help='Absolute path to labels file.')
    parser.add_argument( '-m', '--mean', metavar = 'NUMPY_MEAN_FILE', 
                        type=str, default = None,
                        help = 'Network Numpy mean file.')
    parser.add_argument( '-i', '--image', metavar = 'IMAGE_FILE', 
                        type=str, default = '../../data/images/nps_electric_guitar.png',
                        help = 'Absolute path to image file.')
    parser.add_argument( '-t', '--top', metavar = 'NUM_RESULTS_TO_SHOW', 
                        type=int, default = 1,
                        help = 'Number of inference results to output.')
    return parser


def display_info(input_shape, output_shape, image, ir, labels, mean):
    print()
    print(YELLOW + 'Starting application...' + NOCOLOR)
    print('   - ' + YELLOW + 'Plugin:       ' + NOCOLOR + 'Myriad')
    print('   - ' + YELLOW + 'IR File:     ' + NOCOLOR, ir)
    print('   - ' + YELLOW + 'Input Shape: ' + NOCOLOR, input_shape)
    print('   - ' + YELLOW + 'Output Shape:' + NOCOLOR, output_shape)
    print('   - ' + YELLOW + 'Labels File: ' + NOCOLOR, labels)
    print('   - ' + YELLOW + 'Mean File:   ' + NOCOLOR, mean)
    print('   - ' + YELLOW + 'Image File:   ' + NOCOLOR, image)

    

def infer(image = '../../data/images/nps_electric_guitar.png', ir = '../../caffe/SqueezeNet/squeezenet_v1.0.xml', labels = '../../data/ilsvrc12/synset_words.txt', mean = None, top = 1):

    ####################### 1. Setup Plugin and Network #######################
    # Select the myriad plugin and IRs to be used
    ie = IECore()
    net = IENetwork(model = ir, weights = ir[:-3] + 'bin')

    # Set up the input and output blobs
    input_blob = next(iter(net.inputs))
    output_blob = next(iter(net.outputs))
    input_shape = net.inputs[input_blob].shape
    output_shape = net.outputs[output_blob].shape
    
    # Display model information
    display_info(input_shape, output_shape, image, ir, labels, mean)
    
    # Load the network and get the network shape information
    exec_net = ie.load_network(network = net, device_name = DEVICE)
    n, c, h, w = input_shape

    # Prepare Categories for age and gender networks
    with open(labels) as labels_file:
	    label_list = labels_file.read().splitlines()
	
    ####################### 2. Image Preprocessing #######################
    # Read in the image.
    img = cv2.imread(image)

    if img is None:
        print(RED + "\nUnable to read the image file." + NOCOLOR)
        quit()
			
    # Image preprocessing
    img = cv2.resize(img, (h, w))
    img = img.astype(numpy.float32)

    # Load the mean file and subtract the mean then tranpose the image (hwc to chw)
    if mean is not None:
        ilsvrc_mean = numpy.load(mean).mean(1).mean(1) 
        img[:,:,0] = (img[:,:,0] - ilsvrc_mean[0])
        img[:,:,1] = (img[:,:,1] - ilsvrc_mean[1])
        img[:,:,2] = (img[:,:,2] - ilsvrc_mean[2])
    transposed_img = numpy.transpose(img, (2,0,1))
    reshaped_img = transposed_img.reshape((n, c, h, w))

    ####################### 3. Run the inference #######################
    res = exec_net.infer({input_blob: reshaped_img})
    
    ####################### 4. Process and print the results #######################
    top_ind = numpy.argsort(res[output_blob], axis=1)[0, -top:][::-1]
    print(YELLOW + '\n **********' + NOCOLOR + '  Results  ' + YELLOW + '***********' + NOCOLOR)
    result = ''
    for k, i in enumerate(top_ind):
        result = result + (' Prediction is ' + "%3.1f%%" % (100*res[output_blob][0, i]) + " " + label_list[int(i)] + "\n")
    print('')

    return result, image
    
    
if __name__ == "__main__":
    ARGS = parse_args().parse_args()
    result, image = infer(ARGS.image, ARGS.ir, ARGS.labels, ARGS.mean, ARGS.top)
    print(result)
