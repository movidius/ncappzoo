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

AGE_OUTPUT_LAYER = 'age_conv3'
GENDER_OUTPUT_LAYER = 'prob'
GENDER_LABEL_LIST = ['Female', 'Male']
    
try:
    from openvino.inference_engine import IENetwork, IECore
except:
    print(RED + '\nPlease make sure your OpenVINO environment variables are set by sourcing the' + YELLOW + ' setupvars.sh ' + RED + 'script found in <your OpenVINO install location>/bin/ folder.\n' + NOCOLOR)
    exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description = 'Image classifier using \
                         Intel® Neural Compute Stick 2.' )
    parser.add_argument( '--ir', metavar = 'AGE_GENDER_IR',
                        type=str, default = './age-gender-recognition-retail-0013.xml',
                        help = 'Absolute path to the AgeGender IR file.')
    parser.add_argument( '-i', '--image', metavar = 'IMAGE_FILE', 
                        type=str, default = 'cropped_face_0.png',
                        help = 'Absolute path to image file.')

    return parser


def display_info(input_shape, age_output_shape, gender_output_shape, image, ir):
    print()
    print(YELLOW + 'AgeNet: Starting application...' + NOCOLOR)
    print('   - ' + YELLOW + 'Plugin:      ' + NOCOLOR + ' Myriad')
    print('   - ' + YELLOW + 'IR File:     ' + NOCOLOR, ir)
    print('   - ' + YELLOW + 'Input Shape: ' + NOCOLOR, input_shape)
    print('   - ' + YELLOW + 'Age Output Shape:' + NOCOLOR, age_output_shape)
    print('   - ' + YELLOW + 'Gender Output Shape:' + NOCOLOR, gender_output_shape)


    

def infer():
    ARGS = parse_args().parse_args()
    ir = ARGS.ir
    image = ARGS.image
    ####################### 1. Setup Plugin and Network #######################
    # Select the myriad plugin and IRs to be used
    ie = IECore()
    
    age_gender_net = IENetwork(model = ir, weights = ir[:-3] + 'bin')

    # Set up the input and output blobs
    age_gender_input_blob = next(iter(age_gender_net.inputs))
    # Make sure to select the age gender output
    age_output_blob = AGE_OUTPUT_LAYER
    gender_output_blob = GENDER_OUTPUT_LAYER
    
    age_gender_input_shape = age_gender_net.inputs[age_gender_input_blob].shape
    # output shapes
    age_output_shape = age_gender_net.outputs[age_output_blob].shape
    gender_output_shape = age_gender_net.outputs[gender_output_blob].shape
    
    # Display model information
    display_info(age_gender_input_shape, age_output_shape, gender_output_shape, image, ir)
    
    # Load the network and get the network shape information
    age_gender_exec_net = ie.load_network(network = age_gender_net, device_name = "MYRIAD")
    
    age_gender_n, age_gender_c, age_gender_h, age_gender_w = age_gender_input_shape
    

    ####################### 2. Image Preprocessing #######################
    # Read in the image.

    frame = cv2.imread(image)

    if frame is None:
        print(RED + "\nUnable to read the image file." + NOCOLOR)
        quit()
			
    # Image preprocessing
    image_to_classify = cv2.resize(frame, (age_gender_w, age_gender_h))
    image_to_classify = numpy.transpose(image_to_classify, (2, 0, 1))
    image_to_classify = image_to_classify.reshape((age_gender_n, age_gender_c, age_gender_h, age_gender_w))

    ####################### 3. Run the inference #######################
    cur_request_id = 0
    detection_threshold = 0.5
    
    age_gender_exec_net.start_async(request_id=cur_request_id, inputs={age_gender_input_blob: image_to_classify})
    if age_gender_exec_net.requests[cur_request_id].wait(-1) == 0:
        age_res = age_gender_exec_net.requests[cur_request_id].outputs[age_output_blob]
        age_res = age_res.flatten()
        gender_res = age_gender_exec_net.requests[cur_request_id].outputs[gender_output_blob]
        gender_res = gender_res.flatten()
        top_ind = numpy.argsort(gender_res)[::-1][:1]
        print('\n Gender prediction is ' + "%3.1f%%" % (100*gender_res[top_ind]) + " " + GENDER_LABEL_LIST[int(top_ind)])
        print(" Age prediction is " + str(int(age_res[0]* 100)) + " years old.\n")
        
    
if __name__ == "__main__":
    infer()

