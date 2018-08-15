#!/usr/bin/env python3

# ****************************************************************************
# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.
# ****************************************************************************

# Utilities to help deserialize the output list from
# Intel® Movidius™ Neural Compute Stick (NCS) 

# ---- Deserialize the output from an SSD based network ----
# @param output The NCS returns a list/array in this structure:
# First float16: Number of detections
# Next 6 values: Unused
# Next consecutive batch of 7 values: Detection values
#   0: Image ID (always 0)
#   1: Class ID (index into labels.txt)
#   2: Detection score
#   3: Box left coordinate (x1) - scaled value between 0 & 1
#   4: Box top coordinate (y1) - scaled value between 0 & 1
#   5: Box right coordinate (x2) - scaled value between 0 & 1
#   6: Box bottom coordinate (y2) - scaled value between 0 & 1
#
# @return output_dict A Python dictionary with the following keys:
# output_dict['num_detections'] = Total number of valid detections
# output_dict['detection_classes_<X>'] = Class ID of the detected object
# output_dict['detection_scores_<X>'] = Percetage of the confidance
# output_dict['detection_boxes_<X>'] = A list of 2 tuples [(x1, y1) (x2, y2)]
# Where <X> is a zero-index count of num_detections

import numpy as np

def ssd( output, confidance_threshold, shape ):

    # Dictionary where the deserialized output will be stored
    output_dict = {}

    # Extract the original image's shape
    height, width, channel = shape

    # Total number of detections
    output_dict['num_detections'] = int( output[0] )

    # Variable to track number of valid detections
    valid_detections = 0

    for detection in range( output_dict['num_detections'] ):

        # Skip the first 7 values, and point to the next batch of 7 values
        base_index = 7 + ( 7 * detection )

        # Record only those detections whose confidance meets our threshold
        if( output[ base_index + 2 ] > confidance_threshold ):

            output_dict['detection_classes_' + str(valid_detections)] = \
                int( output[base_index + 1] )

            output_dict['detection_scores_' + str(valid_detections)] = \
                int( output[base_index + 2] * 100 )

            x = [ int( output[base_index + 3] * width ), 
                  int( output[base_index + 5] * width ) ]

            y = [ int( output[base_index + 4] * height ), 
                  int( output[base_index + 6] * height ) ]

            output_dict['detection_boxes_' + str(valid_detections)] = \
                list( zip( y, x ) )

            valid_detections += 1

    # Update total number of detections to valid detections
    output_dict['num_detections'] = int( valid_detections )

    return( output_dict )

# ====== Tiny Yolo ======

# Interpret the output from a single inference of TinyYolo (GetResult)
# and filter out objects/boxes with low probabilities.
# output is the array of floats returned from the API GetResult but converted
# to float32 format.
# input_image_width is the width of the input image
# input_image_height is the height of the input image
# Returns a list of lists. each of the inner lists represent one found object and contain
# the following 6 values:
#    string that is network classification ie 'cat', or 'chair' etc
#    float value for box center X pixel location within source image
#    float value for box center Y pixel location within source image
#    float value for box width in pixels within source image
#    float value for box height in pixels within source image
#    float value that is the probability for the network classification.
def filter_objects(inference_result, input_image_width, input_image_height):

    # the raw number of floats returned from the inference (GetResult())
    num_inference_results = len(inference_result)

    # the 20 classes this network was trained on
    network_classifications = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                               "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                               "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

    # only keep boxes with probabilities greater than this
    probability_threshold = 0.07

    num_classifications = len(network_classifications) # should be 20
    grid_size = 7 # the image is a 7x7 grid.  Each box in the grid is 64x64 pixels
    boxes_per_grid_cell = 2 # the number of boxes returned for each grid cell

    # grid_size is 7 (grid is 7x7)
    # num classifications is 20
    # boxes per grid cell is 2
    all_probabilities = np.zeros((grid_size, grid_size, boxes_per_grid_cell, num_classifications))

    # classification_probabilities  contains a probability for each classification for
    # each 64x64 pixel square of the grid.  The source image contains
    # 7x7 of these 64x64 pixel squares and there are 20 possible classifications
    classification_probabilities = \
        np.reshape(inference_result[0:980], (grid_size, grid_size, num_classifications))
    num_of_class_probs = len(classification_probabilities)

    # The probability scale factor for each box
    box_prob_scale_factor = np.reshape(inference_result[980:1078], (grid_size, grid_size, boxes_per_grid_cell))

    # get the boxes from the results and adjust to be pixel units
    all_boxes = np.reshape(inference_result[1078:], (grid_size, grid_size, boxes_per_grid_cell, 4))
    boxes_to_pixel_units(all_boxes, input_image_width, input_image_height, grid_size)

    # adjust the probabilities with the scaling factor
    for box_index in range(boxes_per_grid_cell): # loop over boxes
        for class_index in range(num_classifications): # loop over classifications
            all_probabilities[:,:,box_index,class_index] = np.multiply(classification_probabilities[:,:,class_index],box_prob_scale_factor[:,:,box_index])


    probability_threshold_mask = np.array(all_probabilities>=probability_threshold, dtype='bool')
    box_threshold_mask = np.nonzero(probability_threshold_mask)
    boxes_above_threshold = all_boxes[box_threshold_mask[0],box_threshold_mask[1],box_threshold_mask[2]]
    classifications_for_boxes_above = np.argmax(all_probabilities,axis=3)[box_threshold_mask[0],box_threshold_mask[1],box_threshold_mask[2]]
    probabilities_above_threshold = all_probabilities[probability_threshold_mask]

    # sort the boxes from highest probability to lowest and then
    # sort the probabilities and classifications to match
    argsort = np.array(np.argsort(probabilities_above_threshold))[::-1]
    boxes_above_threshold = boxes_above_threshold[argsort]
    classifications_for_boxes_above = classifications_for_boxes_above[argsort]
    probabilities_above_threshold = probabilities_above_threshold[argsort]


    # get mask for boxes that seem to be the same object
    duplicate_box_mask = get_duplicate_box_mask(boxes_above_threshold)

    # update the boxes, probabilities and classifications removing duplicates.
    boxes_above_threshold = boxes_above_threshold[duplicate_box_mask]
    classifications_for_boxes_above = classifications_for_boxes_above[duplicate_box_mask]
    probabilities_above_threshold = probabilities_above_threshold[duplicate_box_mask]

    classes_boxes_and_probs = []
    for i in range(len(boxes_above_threshold)):
        classes_boxes_and_probs.append([classifications_for_boxes_above[i],boxes_above_threshold[i][0],boxes_above_threshold[i][1],boxes_above_threshold[i][2],boxes_above_threshold[i][3],probabilities_above_threshold[i]])

    return classes_boxes_and_probs

# creates a mask to remove duplicate objects (boxes) and their related probabilities and classifications
# that should be considered the same object.  This is determined by how similar the boxes are
# based on the intersection-over-union metric.
# box_list is as list of boxes (4 floats for centerX, centerY and Length and Width)
def get_duplicate_box_mask(box_list):
    # The intersection-over-union threshold to use when determining duplicates.
    # objects/boxes found that are over this threshold will be
    # considered the same object
    max_iou = 0.35

    box_mask = np.ones(len(box_list))

    for i in range(len(box_list)):
        if box_mask[i] == 0: continue
        for j in range(i + 1, len(box_list)):
            if get_intersection_over_union(box_list[i], box_list[j]) > max_iou:
                box_mask[j] = 0.0

    filter_iou_mask = np.array(box_mask > 0.0, dtype='bool')
    return filter_iou_mask

# Converts the boxes in box list to pixel units
# assumes box_list is the output from the box output from
# the tiny yolo network and is [grid_size x grid_size x 2 x 4].
def boxes_to_pixel_units(box_list, image_width, image_height, grid_size):

    # number of boxes per grid cell
    boxes_per_cell = 2

    # setup some offset values to map boxes to pixels
    # box_offset will be [[ [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]] ...repeated for 7 ]
    box_offset = np.transpose(np.reshape(np.array([np.arange(grid_size)]*(grid_size*2)),(boxes_per_cell,grid_size, grid_size)),(1,2,0))

    # adjust the box center
    box_list[:,:,:,0] += box_offset
    box_list[:,:,:,1] += np.transpose(box_offset,(1,0,2))
    box_list[:,:,:,0:2] = box_list[:,:,:,0:2] / (grid_size * 1.0)

    # adjust the lengths and widths
    box_list[:,:,:,2] = np.multiply(box_list[:,:,:,2],box_list[:,:,:,2])
    box_list[:,:,:,3] = np.multiply(box_list[:,:,:,3],box_list[:,:,:,3])

    #scale the boxes to the image size in pixels
    box_list[:,:,:,0] *= image_width
    box_list[:,:,:,1] *= image_height
    box_list[:,:,:,2] *= image_width
    box_list[:,:,:,3] *= image_height


# Evaluate the intersection-over-union for two boxes
# The intersection-over-union metric determines how close
# two boxes are to being the same box.  The closer the boxes
# are to being the same, the closer the metric will be to 1.0
# box_1 and box_2 are arrays of 4 numbers which are the (x, y)
# points that define the center of the box and the length and width of
# the box.
# Returns the intersection-over-union (between 0.0 and 1.0)
# for the two boxes specified.
def get_intersection_over_union(box_1, box_2):

    # one diminsion of the intersecting box
    intersection_dim_1 = min(box_1[0]+0.5*box_1[2],box_2[0]+0.5*box_2[2])-\
                         max(box_1[0]-0.5*box_1[2],box_2[0]-0.5*box_2[2])

    # the other dimension of the intersecting box
    intersection_dim_2 = min(box_1[1]+0.5*box_1[3],box_2[1]+0.5*box_2[3])-\
                         max(box_1[1]-0.5*box_1[3],box_2[1]-0.5*box_2[3])

    if intersection_dim_1 < 0 or intersection_dim_2 < 0 :
        # no intersection area
        intersection_area = 0
    else :
        # intersection area is product of intersection dimensions
        intersection_area =  intersection_dim_1*intersection_dim_2

    # calculate the union area which is the area of each box added
    # and then we need to subtract out the intersection area since
    # it is counted twice (by definition it is in each box)
    union_area = box_1[2]*box_1[3] + box_2[2]*box_2[3] - intersection_area;

    # now we can return the intersection over union
    iou = intersection_area / union_area

    return iou

def tinyyolo( output, confidance_threshold, shape, network_width=448, network_height=448 ):

    # Dictionary where the deserialized output will be stored
    output_dict = {}

    # Extract the original image's shape
    height, width, channel = shape

    # filter out all the objects/boxes that don't meet thresholds
    filtered_objects = filter_objects(output.astype(np.float32), network_width, network_height) # fc27 instead of fc12 for yolo_small

    x_ratio = float(width) / network_width
    y_ratio = float(height) / network_height
# @return output_dict A Python dictionary with the following keys:
# output_dict['num_detections'] = Total number of valid detections
    output_dict['num_detections'] = len(filtered_objects)

    # Variable to track number of valid detections
    valid_detections = 0

    for obj_index in range(len(filtered_objects)):
        center_x = int(filtered_objects[obj_index][1] * x_ratio)
        center_y = int(filtered_objects[obj_index][2] * y_ratio)
        half_width = int(filtered_objects[obj_index][3] * x_ratio)//2
        half_height = int(filtered_objects[obj_index][4] * y_ratio)//2

        # calculate box (left, top) and (right, bottom) coordinates
        box_left = max(center_x - half_width, 0)
        box_top = max(center_y - half_height, 0)
        box_right = min(center_x + half_width, width)
        box_bottom = min(center_y + half_height, height)

        output_dict['detection_classes_' + str(valid_detections)] = filtered_objects[obj_index][0]
        output_dict['detection_scores_' + str(valid_detections)] = filtered_objects[obj_index][5] * 100
        output_dict['detection_boxes_' + str(valid_detections)] = [(box_top, box_left), (box_bottom, box_right)]

        valid_detections += 1

    return( output_dict )


# ==== End of file ===========================================================
