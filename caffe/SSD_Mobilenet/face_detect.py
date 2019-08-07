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

try:
    from openvino.inference_engine import IENetwork, IECore
except:
    print(RED + '\nPlease make sure your OpenVINO environment variables are set by sourcing the' + YELLOW + ' setupvars.sh ' + RED + 'script found in <your OpenVINO install location>/bin/ folder.\n' + NOCOLOR)
    exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description = 'Image classifier using \
                         Intel® Neural Compute Stick 2.' )
    parser.add_argument( '--face_ir', metavar = 'FACE_DETECTION_IR_File',
                        type=str, default = './face-detection-adas-0001-fp16.xml',
                        help = 'Absolute path to the face detection neural network IR file.')
    parser.add_argument( '-i', '--image', metavar = 'IMAGE_FILE', 
                        type=str, default = 'image.jpg',
                        help = 'Absolute path to image file.')
    parser.add_argument( '--crop', metavar = 'place to save images',
                        type=str, default = None,
                        help = 'Crop faces from image?.')
    parser.add_argument( '--show',  metavar = 'yes/no',
                        type=str, default = 'no',
                        help = 'Display window.')
    return parser


def display_info(input_shape, output_shape, image, ir, labels, crop, show):
    print()
    print(YELLOW + 'face-detection-retail-0004: Starting application...' + NOCOLOR)
    print('   - ' + YELLOW + 'Plugin:      ' + NOCOLOR + ' Myriad')
    print('   - ' + YELLOW + 'IR File:     ' + NOCOLOR, ir)
    print('   - ' + YELLOW + 'Input Shape: ' + NOCOLOR, input_shape)
    print('   - ' + YELLOW + 'Output Shape:' + NOCOLOR, output_shape)
    print('   - ' + YELLOW + 'Labels File: ' + NOCOLOR, labels)
    print('   - ' + YELLOW + 'Image File:  ' + NOCOLOR, image)
    print('   - ' + YELLOW + 'Crop path:   ' + NOCOLOR, crop)
    print('   - ' + YELLOW + 'Show window: ' + NOCOLOR, show)

    
def infer():
    ARGS = parse_args().parse_args()
    face_ir = ARGS.face_ir
    image = ARGS.image
    crop = ARGS.crop
    show_display = ARGS.show

    faces_found = False
    cur_request_id = 0
    detection_threshold = 0.5
    box_color = (0, 255, 0)
    box_thickness = 1
    
    ####################### 1. Setup Plugin and Network #######################
    # Select the myriad plugin and IRs to be used
    ie = IECore()

    face_net = IENetwork(model = face_ir, weights = face_ir[:-3] + 'bin')

    # Get the input and output node names
    face_input_blob = next(iter(face_net.inputs))
    face_output_blob = next(iter(face_net.outputs))
    
    # Get the input and output shapes from the input/output nodes
    face_input_shape = face_net.inputs[face_input_blob].shape
    face_output_shape = face_net.outputs[face_output_blob].shape
    face_n, face_c, face_h, face_w = face_input_shape
    face_x, face_y, face_detections_count, face_detections_size = face_output_shape
        
    # Display model information
    display_info(face_input_shape, face_output_shape, image, face_ir, None, crop, show_display)
    
    # Load the network and get the network shape information
    face_exec_net = ie.load_network(network = face_net, device_name = "MYRIAD")
    
    ####################### 2. Image Preprocessing #######################
    # Read in the image
    frame = cv2.imread(image)
    image_w = frame.shape[1]
    image_h = frame.shape[0]
    
    if frame is None:
        print(RED + "\nUnable to read the image file." + NOCOLOR)
        quit()
			
    # Image preprocessing
    image_to_classify = cv2.resize(frame, (face_w, face_h))
    image_to_classify = numpy.transpose(image_to_classify, (2, 0, 1))
    image_to_classify = image_to_classify.reshape((face_n, face_c, face_h, face_w))

    ####################### 3. Run the inference #######################
    # queue the inference
    face_exec_net.start_async(request_id=cur_request_id, inputs={face_input_blob: image_to_classify})
    
    # wait for inference to complete
    if face_exec_net.requests[cur_request_id].wait(-1) == 0:
        # get the inference result
        inference_results = face_exec_net.requests[cur_request_id].outputs[face_output_blob]
        for face_num, detection_result in enumerate(inference_results[0][0]):
            # Draw only detection_resultects when probability more than specified threshold
            if detection_result[2] > detection_threshold:
                box_left = int(detection_result[3] * image_w)
                box_top = int(detection_result[4] * image_h)
                box_right = int(detection_result[5] * image_w)
                box_bottom = int(detection_result[6] * image_h)
                class_id = int(detection_result[1])
                
                if crop is not None:
                    faces_found = True
                    cropped_face = frame[box_top:box_bottom, box_left:box_right]
                    cv2.imwrite(crop +"/cropped_face_" + str(face_num) + ".png", cropped_face)
                
                cv2.rectangle(frame, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)
                
    if show_display == 'yes':
        cv2.imshow("face detection retail 0004", frame)
        print("Press any key to continue.")
        while(True):
	        rawkey = cv2.waitKey()
	        if rawkey != -1:
	            break

    if crop is not None and faces_found:
        print("\n Cropped faces saved to " + crop + " folder.\n")
    
    
if __name__ == "__main__":
    infer()
    print("Finished.")
 	
