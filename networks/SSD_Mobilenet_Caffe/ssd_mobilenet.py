#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory. 

import sys
import numpy
import cv2
import argparse
import os

GREEN = '\033[1;32m'
RED = '\033[1;31m'
NOCOLOR = '\033[0m'
YELLOW = '\033[1;33m'

SSD_WINDOW_NAME = "SSD Mobilenet Caffe VOC"

try:
    from openvino.inference_engine import IENetwork, IECore
    import openvino.inference_engine.ie_api
except:
    print(RED + '\nPlease make sure your OpenVINO environment variables are set by sourcing the' + YELLOW + ' setupvars.sh ' + RED + 'script found in <your OpenVINO install location>/bin/ folder.\n' + NOCOLOR)
    exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description = 'Image classifier using \
                         Intel® Neural Compute Stick 2.' )
    parser.add_argument( '--ir', metavar = 'SSD_IR_FILE',
                        type=str, default = './mobilenet-ssd.xml',
                        help = 'Absolute path to the neural network IR file.')
    parser.add_argument( '-i', '--input', metavar = 'IMAGE_FILE', 
                        type=str, default = '../../data/images/nps_chair.png',
                        help = 'input.')
    parser.add_argument( '--labels_file', metavar = 'labels_file',
                        type=str, default = "labels.txt",
                        help = 'labels file for the model.')
    parser.add_argument( '--show',  metavar = 'yes/no',
                        type=str, default = 'yes',
                        help = 'Display window.')
    parser.add_argument( '-d', '--device', metavar = 'DEVICE',
                        type=str, default = 'MYRIAD',
                        help = 'Device to run apps on')
    parser.add_argument( '--dt', metavar = 'detection threshold',
                        type=float, default = 0.50,
                        help = 'Detection threshold')
    return parser


def display_info(input_shape, output_shape, image, ir, labels, show):
    print()
    print(YELLOW + 'Caffe SSD Mobilenet VOC: Starting application...' + NOCOLOR)
    print('   - ' + YELLOW + 'Plugin:      ' + NOCOLOR + ' Myriad')
    print('   - ' + YELLOW + 'IR File:     ' + NOCOLOR, ir)
    print('   - ' + YELLOW + 'Input Shape: ' + NOCOLOR, input_shape)
    print('   - ' + YELLOW + 'Output Shape:' + NOCOLOR, output_shape)
    print('   - ' + YELLOW + 'Labels File: ' + NOCOLOR, labels)
    print('   - ' + YELLOW + 'Image File:  ' + NOCOLOR, image)
    print('   - ' + YELLOW + 'Show window: ' + NOCOLOR, show)


def text_setup(frame, labels_list, class_id, confidence, box_left, box_top):
    # label shape and colorization for displaying
    label_text = labels_list[class_id] + " " + str("{0:.2f}".format(confidence))
    label_background_color = (70, 120, 70) # grayish green background for text
    label_text_color = (255, 255, 255)   # white text
    
    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    label_left = int(box_left)
    label_top = int(box_top) - label_size[1]
    label_right = label_left + label_size[0]
    label_bottom = label_top + label_size[1]

    # set up the greenish colored rectangle background for text
    cv2.rectangle(frame, (label_left - 1, label_top - 5),(label_right + 1, label_bottom + 1), label_background_color, -1)
    # set up text
    cv2.putText(frame, label_text, (int(box_left), int(box_top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)


def infer():
    # parse arguments
    ARGS = parse_args().parse_args()
    # get args
    ir = ARGS.ir
    device = ARGS.device
    show_display = ARGS.show
    labels = ARGS.labels_file
    detection_threshold = ARGS.dt
    # some display parameters
    cur_request_id = 0
    box_color = (0, 255, 0)
    box_thickness = 1
    
    # Window properties
    cv2.namedWindow(SSD_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(SSD_WINDOW_NAME, (640, 360))
    cv2.moveWindow(SSD_WINDOW_NAME, 10, 10)
    # Detemine if using cam or image input
    if ARGS.input == 'cam':
        input_stream = 0
    else:
        input_stream = ARGS.input
        assert os.path.isfile(ARGS.input), "Specified input file doesn't exist"
    
    cap = cv2.VideoCapture(input_stream)
    
    ####################### 1. Setup Plugin and Network #######################
    # Set up the inference engine core and load the IR files
    ie = IECore()
    net = IENetwork(model = ir, weights = ir[:-3] + 'bin')

    # Get the input and output node names
    input_blob = next(iter(net.inputs))
    output_blob = next(iter(net.outputs))
    
    # Get the input and output shapes from the input/output nodes
    input_shape = net.inputs[input_blob].shape
    output_shape = net.outputs[output_blob].shape
    n, c, h, w = input_shape
    x, y, detections_count, detections_size = output_shape
        
    # Display model information
    display_info(input_shape, output_shape, input_stream, ir, labels, show_display)
    
    # Load the network and read a frame
    exec_net = ie.load_network(network = net, device_name = device)
    ret, frame = cap.read()

    print(" Press any key to quit.")
    
    # Process frames, make inference, display results
    while cap.isOpened():
        if frame is None:
            print(RED + "\nUnable to read the input." + NOCOLOR)
            quit()
	    ####################### 2. Preprocessing #######################
        # Image preprocessing
        frame = cv2.flip(frame, 1)
        image_to_classify = cv2.resize(frame, (w, h))
        image_to_classify = numpy.transpose(image_to_classify, (2, 0, 1))
        image_to_classify = image_to_classify.reshape((n, c, h, w))
        # get the capture dimensions, these will be used to scale the bounding box for displaying purposes
        image_w = cap.get(3)
        image_h = cap.get(4)
        # Prepare labels
        with open(labels) as labels_file:
	        labels_list = labels_file.read().splitlines()
	        
        ####################### 3. Run the inference #######################
        # queue the inference
        exec_net.start_async(request_id=cur_request_id, inputs={input_blob: image_to_classify})
        
        # wait for inference to complete
        if exec_net.requests[cur_request_id].wait(-1) == 0:
            # get the inference result
            inference_results = exec_net.requests[cur_request_id].outputs[output_blob]
            # process the results
            for num, detection_result in enumerate(inference_results[0][0]):
                # Draw only detection_resultects when probability more than specified threshold
                if detection_result[2] > detection_threshold:
                    box_left = int(detection_result[3] * image_w)
                    box_top = int(detection_result[4] * image_h)
                    box_right = int(detection_result[5] * image_w)
                    box_bottom = int(detection_result[6] * image_h)
                    class_id = int(detection_result[1])

                    # set up the text to display with the bounding box in the frame
                    text_setup(frame, labels_list, class_id, detection_result[2], box_left, box_top)
                    # set up the detection box in the frame
                    cv2.rectangle(frame, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)
            
            # show the frame in the window
            cv2.imshow(SSD_WINDOW_NAME, frame)
            
            # get another frame from camera if using camera input
            if input_stream == 0:
                key = cv2.waitKey(1)
                if key != -1:
                    cap.release()
                    break
                ret, frame = cap.read()
            else: #  or wait for key press if image input
                while True:
                    key = cv2.waitKey(1)
                    if key != -1:
                        cap.release()
                        break
    # Clean up all windows
    cv2.destroyAllWindows()
    
# main entrypoint for the script
if __name__ == "__main__":
    infer()
    print(" Finished.")
 	
