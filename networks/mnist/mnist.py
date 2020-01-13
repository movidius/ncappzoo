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

NETWORK_DIM = (28, 28)

try:
    from openvino.inference_engine import IENetwork, IEPlugin
except:
    print(RED + 'Please make sure your OpenVINO environment variables are set.' + NOCOLOR)
    exit(1)
    

def parse_args():
    parser = argparse.ArgumentParser(description = 'Image classifier using \
                         Intel® Neural Compute Stick 2.' )
    parser.add_argument( '--ir', metavar = 'IR_File',
                        type=str, default = 'mnist_inference.xml', 
                        help = 'Absolute path to the neural network IR file.')
    parser.add_argument( '-l', '--labels', metavar = 'LABEL_FILE', 
                        type=str, default = 'categories.txt',
                        help='Absolute path to labels file.')
    parser.add_argument( '-m', '--mean', metavar = 'NUMPY_MEAN_FILE', 
                        type=str, default = None,
                        help = 'Network Numpy mean file.')
    parser.add_argument( '-i', '--image', metavar = 'IMAGE_FILE', 
                        type=str, default = '../../data/images/nps_electric_guitar.png',
                        help = 'Absolute path to image file.')
    parser.add_argument( '-n', '--number_top', metavar = 'NUM_RESULTS_TO_SHOW', 
                        type=int, default = 1,
                        help = 'Number of inference results to output.')
    return parser


def display_info(ARGS, input_shape, output_shape):
    print()
    print(YELLOW + 'Starting application...' + NOCOLOR)
    print('   - ' + YELLOW + 'Plugin:       ' + NOCOLOR + 'Myriad')
    print('   - ' + YELLOW + 'IR File:     ' + NOCOLOR, ARGS.ir)
    print('   - ' + YELLOW + 'Input Shape: ' + NOCOLOR, input_shape)
    print('   - ' + YELLOW + 'Output Shape:' + NOCOLOR, output_shape)
    print('   - ' + YELLOW + 'Labels File: ' + NOCOLOR, ARGS.labels)
    print('   - ' + YELLOW + 'Mean File:   ' + NOCOLOR, ARGS.mean)
    print('   - ' + YELLOW + 'Image file:  ' + NOCOLOR, ARGS.image)
    
    
def main():
    ARGS = parse_args().parse_args()

    ####################### 1. Setup Plugin and Network #######################
    # Select the myriad plugin and IRs to be used
    plugin = IEPlugin(device='MYRIAD')
    net = IENetwork(model = ARGS.ir, weights = ARGS.ir[:-3] + 'bin')

    # Set up the input and output blobs
    input_blob = next(iter(net.inputs))
    output_blob = next(iter(net.outputs))

    # Load the network and get the network shape information
    exec_net = plugin.load(network = net)
    input_shape = tuple(net.inputs[input_blob].shape)
    output_shape = net.outputs[output_blob].shape
    display_info(ARGS, input_shape, output_shape)
    # Prepare Categories for age and gender networks
    with open(ARGS.labels) as labels_file:
	    label_list = labels_file.read().splitlines()
	
    ####################### 2. Image Preprocessing #######################
    # Read in the image.
    img = cv2.imread(ARGS.image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Image preprocessing
    img = cv2.resize(img, NETWORK_DIM)
    img = img.astype(numpy.float32)
    img[:] = ((img[:]) * (1.0 / 255.0))
    
    img = img.reshape(input_shape)

    ####################### 3. Run the inference #######################
    res = exec_net.infer({input_blob: img})
    
    ####################### 4. Process and print the results #######################
    top_ind = numpy.argsort(res[output_blob], axis=1)[0, -ARGS.number_top:][::-1]
    print(YELLOW + '\n **********' + NOCOLOR + '  Results  ' + YELLOW + '***********' + NOCOLOR)
    for k, i in enumerate(top_ind):
        print('Prediction is ' + YELLOW + label_list[int(i)] + NOCOLOR + " with a confidence of " + YELLOW + "%3.1f%%." % (100*res[output_blob][0, i]) + NOCOLOR)
    print('')

if __name__ == "__main__":
    sys.exit(main() or 0)

