#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.

import cv2
import numpy as np
import sys
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


IOU_THRESHOLD = 0.1

def parse_args():
    parser = argparse.ArgumentParser(description = 'Tiny Yolo V2 Object Detection using \
                         Intel® Neural Compute Stick 2.' )
    parser.add_argument( '--ir', metavar = 'IR_File',
                        type=str, default = 'yolov2-tiny-voc.xml', 
                        help = 'Absolute path to the neural network IR xml file.')
    parser.add_argument( '-l', '--labels', metavar = 'LABEL_FILE', 
                        type=str, default = 'labels.txt',
                        help='Absolute path to labels file.')
    parser.add_argument( '-i', '--input', metavar = 'INPUT', 
                        type=str, default = '../../data/images/nps_chair.png',
                        help = 'Absolute path to image file.')
    parser.add_argument( '--threshold', metavar = 'FLOAT', 
                        type=float, default = 0.65,
                        help = 'Threshold for detection.')
                      
    return parser


def display_info(input_shape, output_shape, image, ir, labels):
    print()
    print(YELLOW + 'Tiny Yolo v2: Starting application...' + NOCOLOR)
    print('   - ' + YELLOW + 'Plugin:       ' + NOCOLOR + 'Myriad')
    print('   - ' + YELLOW + 'IR File:     ' + NOCOLOR, ir)
    print('   - ' + YELLOW + 'Input Shape: ' + NOCOLOR, input_shape)
    print('   - ' + YELLOW + 'Output Shape:' + NOCOLOR, output_shape)
    print('   - ' + YELLOW + 'Labels File: ' + NOCOLOR, labels)
    print('   - ' + YELLOW + 'Image File:   ' + NOCOLOR, image)


def sigmoid(x):
    return 1.0 / (1 + np.exp(x * -1.0))


def calculate_overlap(x1, w1, x2, w2):
    box1_coordinate = max(x1 - w1 / 2.0, x2 - w2 / 2.0)
    box2_coordinate = min(x1 + w1 / 2.0, x2 + w2 / 2.0)
    overlap = box2_coordinate - box1_coordinate
    return overlap


def calculate_iou(box, truth):
    # calculate the iou intersection over union by first calculating the overlapping height and width
    width_overlap = calculate_overlap(box[0], box[2], truth[0], truth[2])
    height_overlap = calculate_overlap(box[1], box[3], truth[1], truth[3])
    # no overlap
    if width_overlap < 0 or height_overlap < 0:
        return 0

    intersection_area = width_overlap * height_overlap
    union_area = box[2] * box[3] + truth[2] * truth[3] - intersection_area
    iou = intersection_area / union_area
    return iou


def apply_nms(boxes):
    # sort the boxes by score in descending order
    sorted_boxes = sorted(boxes, key=lambda d: d[7])[::-1]
    high_iou_objs = dict()
    # compare the iou for each of the detected objects
    for current_object in range(len(sorted_boxes)):
        if current_object in high_iou_objs:
            continue

        truth = sorted_boxes[current_object]
        for next_object in range(current_object + 1, len(sorted_boxes)):
            if next_object in high_iou_objs:
                continue
            box = sorted_boxes[next_object]
            iou = calculate_iou(box, truth)
            if iou >= IOU_THRESHOLD:
                high_iou_objs[next_object] = 1

    # filter and sort detected items
    filtered_result = list()
    for current_object in range(len(sorted_boxes)):
        if current_object not in high_iou_objs:
            filtered_result.append(sorted_boxes[current_object])
    return filtered_result


def post_processing(output, original_img, label_list, threshold):

    num_classes = 20
    num_grids = 13
    num_anchor_boxes = 5
    original_results = output.astype(np.float32)   

    # Tiny Yolo V2 uses a 13 x 13 grid with 5 anchor boxes for each grid cell.
    # This specific model was trained with the VOC Pascal data set and is comprised of 20 classes

    original_results = np.reshape(original_results, (5, 25, 13, 13))
    reordered_results = np.transpose(original_results, (2, 3, 0, 1))
    reordered_results = np.reshape(reordered_results, (169, 5, 25))

    # The 125 results need to be re-organized into 5 chunks of 25 values
    # 20 classes + 1 score + 4 coordinates = 25 values
    # 25 values for each of the 5 anchor bounding boxes = 125 values
    #reordered_results = np.zeros((13 * 13, 5, 25))

    index = 0
    #for row in range( num_grids ):
    #    for col in range( num_grids ):
    #        for b_box_voltron in range(125):
    #            b_box = row * num_grids + col
    #            b_box_num = int(b_box_voltron / 25)
    #            b_box_info = b_box_voltron % 25
    #            reordered_results[b_box][b_box_num][b_box_info] = original_results[row][col][b_box_voltron]

    # shapes for the 5 Tiny Yolo v2 bounding boxes
    anchor_boxes = [1.08,1.19, 3.42,4.41, 6.63,11.38, 9.42,5.11, 16.62,10.52]

    boxes = list()
    # iterate through the grids and anchor boxes and filter out all scores which do not exceed the DETECTION_THRESHOLD
    for row in range(num_grids):
        for col in range(num_grids):
            for anchor_box_num in range(num_anchor_boxes):
                box = list()
                class_list = list()
                current_score_total = 0
                # calculate the coordinates for the current anchor box
                box_x = (col + sigmoid(reordered_results[row * 13 + col][anchor_box_num][0])) / 13.0
                box_y = (row + sigmoid(reordered_results[row * 13 + col][anchor_box_num][1])) / 13.0
                box_w = (np.exp(reordered_results[row * 13 + col][anchor_box_num][2]) *
                         anchor_boxes[2 * anchor_box_num]) / 13.0
                box_h = (np.exp(reordered_results[row * 13 + col][anchor_box_num][3]) *
                         anchor_boxes[2 * anchor_box_num + 1]) / 13.0
                
                # find the class with the highest score
                for class_enum in range(num_classes):
                    class_list.append(reordered_results[row * 13 + col][anchor_box_num][5 + class_enum])



                current_score_total = sum(class_list)
                for current_class in range(len(class_list)):
                    class_list[current_class] = class_list[current_class] * 1.0 / current_score_total

                # probability that the current anchor box contains an item
                object_confidence = sigmoid(reordered_results[row * 13 + col][anchor_box_num][4])
                # highest class score detected for the object in the current anchor box
                highest_class_score = max(class_list)
                # index of the class with the highest score
                class_w_highest_score = class_list.index(max(class_list)) + 1
                # the final score for the detected object
                final_object_score = object_confidence * highest_class_score

                box.append(box_x)
                box.append(box_y)
                box.append(box_w)
                box.append(box_h)
                box.append(class_w_highest_score)
                box.append(object_confidence)
                box.append(highest_class_score)
                box.append(final_object_score)

                # filter out all detected objects with a score less than the threshold
                if final_object_score > threshold:
                    boxes.append(box)

    # gets rid of all duplicate boxes using non-maximal suppression
    results = apply_nms(boxes)

    image_width = original_img.shape[1]
    image_height = original_img.shape[0]

    # calculate the actual box coordinates in relation to the input image
    print('\n Found this many objects in the image: ' + str(len(results)))
    for box in results:
        box_xmin = int((box[0] - box[2] / 2.0) * image_width)
        box_xmax = int((box[0] + box[2] / 2.0) * image_width)
        box_ymin = int((box[1] - box[3] / 2.0) * image_height)
        box_ymax = int((box[1] + box[3] / 2.0) * image_height)
        # ensure the box is not drawn out of the window resolution
        if box_xmin < 0:
            box_xmin = 0
        if box_xmax > image_width:
            box_xmax = image_width
        if box_ymin < 0:
            box_ymin = 0
        if box_ymax > image_height:
            box_ymax = image_height

        print(" - object: " + YELLOW + label_list[box[4]] + NOCOLOR + " is at left: " + str(box_xmin) + " top: " + str(box_ymin) + " right: " + str(box_xmax) + " bottom: " + str(box_ymax))

        # label shape and colorization
        label_text = label_list[box[4]] + " " + str("{0:.2f}".format(box[5]*box[6]))
        label_background_color = (70, 120, 70) # grayish green background for text
        label_text_color = (255, 255, 255)   # white text

        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_left = int(box_xmin)
        label_top = int(box_ymin) - label_size[1]
        label_right = label_left + label_size[0]
        label_bottom = label_top + label_size[1]

        # set up the colored rectangle background for text
        cv2.rectangle(original_img, (label_left - 1, label_top - 5),(label_right + 1, label_bottom + 1),
                      label_background_color, -1)
        # set up text
        cv2.putText(original_img, label_text, (int(box_xmin), int(box_ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    label_text_color, 1)
        # set up the rectangle around the object
        cv2.rectangle(original_img, (int(box_xmin), int(box_ymin)), (int(box_xmax), int(box_ymax)), (0, 255, 0), 2)

    # display all items overlayed in the render window
    cv2.imshow('Tiny Yolo V2 Tensorflow VOC (hit any key to exit)', original_img)
    cv2.waitKey()


def main():
    ARGS = parse_args().parse_args()
    input_image = ARGS.input
    labels = ARGS.labels
    ir = ARGS.ir
    threshold = ARGS.threshold

    # read an image in bgr format
    img = cv2.imread(input_image)
    original_img = img
	
    ie = IECore()
    net = IENetwork(model = ir, weights = ir[:-3] + 'bin')
	
    # Set up the input and output blobs
    input_blob = next(iter(net.inputs))
    output_blob = next(iter(net.outputs))
    input_shape = net.inputs[input_blob].shape
    output_shape = net.outputs[output_blob].shape
    
    # Display model information
    display_info(input_shape, output_shape, input_image, ir, labels)
    
    # Prepare Categories
    with open(labels) as labels_file:
	    label_list = labels_file.read().splitlines()
    
    # Load the network and get the network shape information
    exec_net = ie.load_network(network = net, device_name = DEVICE)
    n, c, h, w = input_shape

    resized_img = cv2.resize(img, (h, w), cv2.INTER_LINEAR)

    # transpose the image to rgb
    transposed_img = np.transpose(resized_img, (2,0,1))
    reshaped_img = transposed_img.reshape((n, c, h, w))

    resized_img = reshaped_img.astype(np.float32)

    # make an inference
    output = exec_net.infer({input_blob: reshaped_img})
    
    print('\n Displaying image with objects detected in GUI...')
    print(' Click in the GUI window and hit any key to exit.')
    
    # Tiny Yolo V2 requires post processing to filter out duplicate objects and low score objects
    # After post processing, the app will display the image and any detected objects
    post_processing(output[output_blob], original_img, label_list, threshold)

    # clean up
    print("\n Finished.")

# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())
