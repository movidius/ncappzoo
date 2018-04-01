#!/usr/bin/python3

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

# ==== End of file ===========================================================
