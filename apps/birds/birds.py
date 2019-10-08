#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.

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


import sys
import numpy as np
import cv2
import os
from googlenet_processor import googlenet_processor
from tiny_yolo_processor import tiny_yolo_processor

# will execute on all images in this directory
input_image_path = './images'

ty_ir= './tiny-yolo-v1_53000.xml'
gn_ir= './googlenet-v1.xml'

# labels to display along with boxes if googlenet classification is good
gn_labels = [""]
cv_window_name = 'Birds - Q to quit or any key to advance'


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

# Displays a gui window with an image that contains
# boxes and lables for found objects.  will not return until
# user presses a key or times out.
# source_image is the original image before resizing or otherwise changed
#
# filtered_objects is a list of lists (as returned from filter_objects()
#   and then added to by get_googlenet_classifications()
#   each of the inner lists represent one found object and contain
#   the following values:
#     string that is yolo network classification ie 'bird'
#     float value for box center X pixel location within source image
#     float value for box center Y pixel location within source image
#     float value for box width in pixels within source image
#     float value for box height in pixels within source image
#     float value that is the probability for the yolo classification.
#     int value that is the index of the googlenet classification
#     string value that is the googlenet classification string.
#     float value that is the googlenet probability
#
# Returns true if should go to next image or false if
# should not.
def display_objects_in_gui(source_image, filtered_objects, ty_processor):

    DISPLAY_BOX_WIDTH_PAD = 0
    DISPLAY_BOX_HEIGHT_PAD = 20

    # if googlenet returns a probablity less than this then
    # just use the tiny yolo more general classification ie 'bird'
    GOOGLE_PROB_MIN = 0.5

	# copy image so we can draw on it.
    display_image = source_image.copy()
    source_image_width = source_image.shape[1]
    source_image_height = source_image.shape[0]

    x_ratio = float(source_image_width) / ty_processor.ty_w
    y_ratio = float(source_image_height) / ty_processor.ty_h

    # loop through each box and draw it on the image along with a classification label
    for obj_index in range(len(filtered_objects)):
        center_x = int(filtered_objects[obj_index][1] * x_ratio)
        center_y = int(filtered_objects[obj_index][2]* y_ratio)
        half_width = int(filtered_objects[obj_index][3]*x_ratio)//2 + DISPLAY_BOX_WIDTH_PAD
        half_height = int(filtered_objects[obj_index][4]*y_ratio)//2 + DISPLAY_BOX_HEIGHT_PAD

        # calculate box (left, top) and (right, bottom) coordinates
        box_left = max(center_x - half_width, 0)
        box_top = max(center_y - half_height, 0)
        box_right = min(center_x + half_width, source_image_width)
        box_bottom = min(center_y + half_height, source_image_height)

        #draw the rectangle on the image.  This is hopefully around the object
        box_color = (0, 255, 0)  # green box
        box_thickness = 2
        cv2.rectangle(display_image, (box_left, box_top),(box_right, box_bottom), box_color, box_thickness)

        # draw the classification label string just above and to the left of the rectangle
        label_background_color = (70, 120, 70) # greyish green background for text
        label_text_color = (255, 255, 255)   # white text

        if (filtered_objects[obj_index][8] > GOOGLE_PROB_MIN):
            label_text = filtered_objects[obj_index][7] + ' : %.2f' % filtered_objects[obj_index][8]
        else:
            label_text = filtered_objects[obj_index][0] + ' : %.2f' % filtered_objects[obj_index][5]

        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_left = box_left
        label_top = box_top - label_size[1]
        label_right = label_left + label_size[0]
        label_bottom = label_top + label_size[1]
        cv2.rectangle(display_image,(label_left-1, label_top-1),(label_right+1, label_bottom+1), label_background_color, -1)

        # label text above the box
        cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

    # display text to let user know how to quit
    cv2.rectangle(display_image,(0, 0),(140, 30), (128, 128, 128), -1)
    cv2.putText(display_image, "Q to Quit", (10, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    cv2.putText(display_image, "Any key to advance", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    cv2.imshow(cv_window_name, display_image)
    raw_key = cv2.waitKey(3000)
    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
        return False

    return True


# Executes googlenet inferences on all objects defined by filtered_objects
# To run the inferences will crop an image out of source image based on the
# boxes defined in filtered_objects and use that as input for googlenet.
#
#
# source_image the original image on which the inference was run.  The boxes
#   defined by filtered_objects are rectangles within this image and will be
#   used as input for googlenet.  This image may be scaled differently from 
#   the tiny yolo network dimensions in which case the boxes in filtered objects
#   will be scaled to match.
#
# filtered_objects [IN/OUT] upon input is a list of lists (as returned from filter_objects()
#   each of the inner lists represent one found object and contain
#   the following 6 values:
#     string that is network classification ie 'cat', or 'chair' etc
#     float value for box center X pixel location within source image
#     float value for box center Y pixel location within source image
#     float value for box width in pixels within source image
#     float value for box height in pixels within source image
#     float value that is the probability for the network classification.
#   upon output the following 3 values from the googlenet inference will
#   be added to each inner list of filtered_objects
#     int value that is the index of the googlenet classification
#     string value that is the googlenet classification string.
#     float value that is the googlenet probability
#
# returns None
def get_googlenet_classifications(source_image, filtered_objects, gn_processor, ty_processor):

    source_image_width = source_image.shape[1]
    source_image_height = source_image.shape[0]
    x_scale = float(source_image_width) / ty_processor.ty_w
    y_scale = float(source_image_height) / ty_processor.ty_h

    # pad the height and width of the image boxes by this amount
    # to make sure we get the whole object in the image that
    # we pass to googlenet
    WIDTH_PAD = int(20 * x_scale)
    HEIGHT_PAD = int(30 * y_scale)

    # loop through each box and crop the image in that rectangle
    # from the source image and then use it as input for googlenet
    
    for obj_index in range(len(filtered_objects)):
        center_x = int(filtered_objects[obj_index][1]*x_scale)
        center_y = int(filtered_objects[obj_index][2]*y_scale)
        half_width = int(filtered_objects[obj_index][3]*x_scale)//2 + WIDTH_PAD
        half_height = int(filtered_objects[obj_index][4]*y_scale)//2 + HEIGHT_PAD

        # Calculate box (left, top) and (right, bottom) coordinates
        box_left = max(center_x - half_width, 0)
        box_top = max(center_y - half_height, 0)
        box_right = min(center_x + half_width, source_image_width)
        box_bottom = min(center_y + half_height, source_image_height)

        one_image = source_image[box_top:box_bottom, box_left:box_right]
        # Run the googlenet inference
        filtered_objects[obj_index] += gn_processor.googlenet_inference(one_image)


    return


# This function is called from the entry point to do
# all the work.
def main():
    global input_image_filename_list
    print('Running NCS birds example')

    # get list of all the .jpg files in the image directory
    input_image_filename_list = os.listdir(input_image_path)
    input_image_filename_list = [input_image_path + '/' + i for i in input_image_filename_list if i.endswith('.jpg')]

    if (len(input_image_filename_list) < 1):
        # no images to show
        print('No .jpg files found')
        return 1

    print('Q to quit, or any key to advance to next image')

    cv2.namedWindow(cv_window_name)

    # Create Inference Engine Core to manage available devices and their plugins internally.
    ie = IECore()
    # Create Tiny Yolo and GoogLeNet processors for running inferences. 
    # Please see tiny_yolo_processor.py and googlenet_processor.py for more information.
    ty_processor = tiny_yolo_processor(ty_ir, ie, DEVICE)
    gn_processor = googlenet_processor(gn_ir, ie, DEVICE)

    for input_image_file in input_image_filename_list :

        # Read image from file, resize it to network width and height
        # save a copy in display_image for display, then convert to float32, normalize (divide by 255),
        # and finally convert to convert to float16 to pass to LoadTensor as input for an inference
        input_image = cv2.imread(input_image_file)
        
        # resize the image to be a standard width for all images and maintain aspect ratio
        STANDARD_RESIZE_WIDTH = 800
        input_image_width = input_image.shape[1]
        input_image_height = input_image.shape[0]

        standard_scale = float(STANDARD_RESIZE_WIDTH) / input_image_width
        new_width = int(input_image_width * standard_scale) # this should be == STANDARD_RESIZE_WIDTH
        new_height = int(input_image_height * standard_scale)
        input_image = cv2.resize(input_image, (new_width, new_height), cv2.INTER_LINEAR)
        display_image = input_image

        # Run the tiny yolo inference and get a list of filtered objects
        filtered_objs = ty_processor.tiny_yolo_inference(input_image)
        # Pass the list of filtered objects to googlenet to classify
        get_googlenet_classifications(display_image, filtered_objs, gn_processor, ty_processor)

        # check if the window has been closed.  all properties will return -1.0
        # for windows that are closed. If the user has closed the window via the
        # x on the title bar then we will break out of the loop.  we are
        # getting property aspect ratio but it could probably be any property
        # may only work with opencv 3.x
        try:
            prop_asp = cv2.getWindowProperty(cv_window_name, cv2.WND_PROP_ASPECT_RATIO)
        except:
            break
        prop_asp = cv2.getWindowProperty(cv_window_name, cv2.WND_PROP_ASPECT_RATIO)
        if (prop_asp < 0.0):
            # the property returned was < 0 so assume window was closed by user
            break

        ret_val = display_objects_in_gui(display_image, filtered_objs, ty_processor)
        if (not ret_val):
            break

    print(' Finished.')


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())
