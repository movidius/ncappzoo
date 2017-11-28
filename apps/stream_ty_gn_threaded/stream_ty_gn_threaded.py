#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.
# NPS

from mvnc import mvncapi as mvnc
import sys
import numpy as np
import cv2
import time
import datetime
import queue
from googlenet_processor import googlenet_processor
from tiny_yolo_processor import tiny_yolo_processor
from camera_processor import camera_processor

# the networks compiled for NCS via ncsdk tools
TINY_YOLO_GRAPH_FILE = './yolo_tiny.graph'
GOOGLENET_GRAPH_FILE = './googlenet.graph'

CAMERA_INDEX = 0
CAMERA_REQUEST_VID_WIDTH = 640
CAMERA_REQUEST_VID_HEIGHT = 480

CAMERA_QUEUE_PUT_WAIT_MAX = 0.001
CAMERA_QUEUE_FULL_SLEEP_SECONDS = 0.01
# for title bar of GUI window
cv_window_name = 'stream_ty_gn_threaded - Q to quit'

CAMERA_QUEUE_SIZE = 1
GN_INPUT_QUEUE_SIZE = 10
GN_OUTPUT_QUEUE_SIZE = 10
TY_OUTPUT_QUEUE_SIZE = 2

# number of seconds to wait when putting or getting from queue's
# besides the camera output queue.
QUEUE_WAIT_MAX = 2

# input and output queueu for the googlenet processor.
gn_input_queue = queue.Queue(GN_INPUT_QUEUE_SIZE)
gn_output_queue = queue.Queue(GN_OUTPUT_QUEUE_SIZE)

ty_proc = None
gn_proc = None

############################################################
# Tuning variables

# if googlenet returns a probablity less than this then
# just use the tiny yolo more general classification ie 'bird'
GN_PROBABILITY_MIN = 0.5

# only keep boxes with probabilities greater than this
# when doing the tiny yolo filtering.  This is only an initial value,
# pressing the B/b keys will adjust up or down respectively
TY_INITIAL_BOX_PROBABILITY_THRESHOLD = 0.10

# The intersection-over-union threshold to use when determining duplicates.
# objects/boxes found that are over this threshold will be considered the
# same object when filtering the Tiny Yolo output.  This is only an initial
# value.  pressing the I/i key will adjust up or down respectively.
TY_INITIAL_MAX_IOU = 0.35


# end of tuning variables
#######################################################


# Displays a gui window with an image that contains
# boxes and lables for found objects.  The
# source_image is the image on which the inference was run.
#
# filtered_objects is a list of lists (as returned from filter_objects()
#   and then added to by get_googlenet_classifications()
#   each of the inner lists represent one found object and contain
#   the following values:
#     [0]:string that is yolo network classification ie 'bird'
#     [1]:float value for box center X pixel location within source image
#     [2]:float value for box center Y pixel location within source image
#     [3]:float value for box width in pixels within source image
#     [4]:float value for box height in pixels within source image
#     [5]:float value that is the probability for the yolo classification.
#     [6]:int value that is the index of the googlenet classification
#     [7]:string value that is the googlenet classification string.
#     [8]:float value that is the googlenet probability
#
# Returns True if should go to next image or False if
# should not.
def overlay_on_image(display_image, filtered_objects):

    DISPLAY_BOX_WIDTH_PAD = 0
    DISPLAY_BOX_HEIGHT_PAD = 20

    source_image_width = display_image.shape[1]
    source_image_height = display_image.shape[0]

    # loop through each box and draw it on the image along with a classification label
    for obj_index in range(len(filtered_objects)):
        center_x = int(filtered_objects[obj_index][1])
        center_y = int(filtered_objects[obj_index][2])
        half_width = int(filtered_objects[obj_index][3])//2 + DISPLAY_BOX_WIDTH_PAD
        half_height = int(filtered_objects[obj_index][4])//2 + DISPLAY_BOX_HEIGHT_PAD

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

        if (filtered_objects[obj_index][8] > GN_PROBABILITY_MIN):
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
    cv2.rectangle(display_image,(0, 0),(100, 15), (128, 128, 128), -1)
    cv2.putText(display_image, "Q to Quit", (10, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)


# Executes googlenet inferences on all objects defined by filtered_objects
# To run the inferences will crop an image out of source image based on the
# boxes defined in filtered_objects and use that as input for googlenet.
#
# gn_graph is the googlenet graph object on which the inference should be executed.
#
# source_image the original image on which the inference was run.  The boxes
#   defined by filtered_objects are rectangles within this image and will be
#   used as input for googlenet
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
def get_googlenet_classifications(source_image, filtered_objects):
    global gn_input_queue, gn_output_queue

    # pad the height and width of the image boxes by this amount
    # to make sure we get the whole object in the image that
    # we pass to googlenet
    WIDTH_PAD = 20
    HEIGHT_PAD = 30

    source_image_width = source_image.shape[1]
    source_image_height = source_image.shape[0]


    # loop through each box and crop the image in that rectangle
    # from the source image and then use it as input for googlenet
    # we are basically gathering the googlenet results
    # serially here on the main thread.  we could have used
    # a separate thread but probably wouldn't give much improvement
    # since tiny yolo is already working on another thread
    for obj_index in range(len(filtered_objects)):
        center_x = int(filtered_objects[obj_index][1])
        center_y = int(filtered_objects[obj_index][2])
        half_width = int(filtered_objects[obj_index][3])//2 + WIDTH_PAD
        half_height = int(filtered_objects[obj_index][4])//2 + HEIGHT_PAD

        # calculate box (left, top) and (right, bottom) coordinates
        box_left = max(center_x - half_width, 0)
        box_top = max(center_y - half_height, 0)
        box_right = min(center_x + half_width, source_image_width)
        box_bottom = min(center_y + half_height, source_image_height)

        # get one image by clipping a box out of source image
        one_image = source_image[box_top:box_bottom, box_left:box_right]

        gn_input_queue.put(one_image, True, QUEUE_WAIT_MAX)

    for obj_index in range(len(filtered_objects)):
        result_list = gn_output_queue.get(True, QUEUE_WAIT_MAX)
        filtered_objects[obj_index] += result_list

    return


# Executes googlenet inferences on all objects defined by filtered_objects
# To run the inferences will crop an image out of source image based on the
# boxes defined in filtered_objects and use that as input for googlenet.
#
# gn_graph is the googlenet graph object on which the inference should be executed.
#
# source_image the original image on which the inference was run.  The boxes
#   defined by filtered_objects are rectangles within this image and will be
#   used as input for googlenet
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
def get_googlenet_classifications_no_queue(gn_proc, source_image, filtered_objects):
    global gn_input_queue, gn_output_queue

    # pad the height and width of the image boxes by this amount
    # to make sure we get the whole object in the image that
    # we pass to googlenet
    WIDTH_PAD = 20
    HEIGHT_PAD = 30

    source_image_width = source_image.shape[1]
    source_image_height = source_image.shape[0]

    # id for all sub images within the larger image
    image_id = datetime.datetime.now().timestamp()


    # loop through each box and crop the image in that rectangle
    # from the source image and then use it as input for googlenet
    for obj_index in range(len(filtered_objects)):
        center_x = int(filtered_objects[obj_index][1])
        center_y = int(filtered_objects[obj_index][2])
        half_width = int(filtered_objects[obj_index][3])//2 + WIDTH_PAD
        half_height = int(filtered_objects[obj_index][4])//2 + HEIGHT_PAD

        # calculate box (left, top) and (right, bottom) coordinates
        box_left = max(center_x - half_width, 0)
        box_top = max(center_y - half_height, 0)
        box_right = min(center_x + half_width, source_image_width)
        box_bottom = min(center_y + half_height, source_image_height)

        # get one image by clipping a box out of source image
        one_image = source_image[box_top:box_bottom, box_left:box_right]

        # Get a googlenet inference on that one image and add the information
        # to the filtered objects list
        filtered_objects[obj_index] += gn_proc.googlenet_inference(one_image, image_id)

    return



# handles key presses by adjusting global thresholds etc.
# raw_key is the return value from cv2.waitkey
# returns False if program should end, or True if should continue
def handle_keys(raw_key):
    global GN_PROBABILITY_MIN, ty_proc, gn_proc
    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
        return False

    elif (ascii_code == ord('B')):
        ty_proc.set_box_probability_threshold(ty_proc.get_box_probability_threshold() + 0.05)
        print("New tiny yolo box probability threshold is " + str(ty_proc.get_box_probability_threshold()))
    elif (ascii_code == ord('b')):
        ty_proc.set_box_probability_threshold(ty_proc.get_box_probability_threshold() - 0.05)
        print("New tiny yolo box probability threshold  is " + str(ty_proc.get_box_probability_threshold()))

    elif (ascii_code == ord('G')):
        GN_PROBABILITY_MIN = GN_PROBABILITY_MIN + 0.05
        print("New GN_PROBABILITY_MIN is " + str(GN_PROBABILITY_MIN))
    elif (ascii_code == ord('g')):
        GN_PROBABILITY_MIN = GN_PROBABILITY_MIN - 0.05
        print("New GN_PROBABILITY_MIN is " + str(GN_PROBABILITY_MIN))

    elif (ascii_code == ord('I')):
        ty_proc.set_max_iou(ty_proc.get_max_iou() + 0.05)
        print("New tiny yolo max IOU is " + str(ty_proc.get_max_iou() ))
    elif (ascii_code == ord('i')):
        ty_proc.set_max_iou(ty_proc.get_max_iou() - 0.05)
        print("New tiny yolo max IOU is " + str(ty_proc.get_max_iou() ))

    return True

# prints information for the user when program starts.
def print_info():
    print('Running stream_ty_gn_threaded')
    print('Keys:')
    print("  'Q'/'q' to Quit")
    print("  'B'/'b' to inc/dec the Tiny Yolo box probability threshold")
    print("  'I'/'i' to inc/dec the Tiny Yolo box intersection-over-union threshold")
    print("  'G'/'g' to inc/dec the GoogLeNet probability threshold")
    print('')


# This function is called from the entry point to do
# all the work.
def main():
    global gn_input_queue, gn_output_queue, ty_proc, gn_proc

    print_info()

    # Set logging level and initialize/open the first NCS we find
    mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 0)
    devices = mvnc.EnumerateDevices()
    if len(devices) < 2:
        print('This application requires two NCS devices.')
        print('Insert two devices and try again!')
        return 1
    ty_device = mvnc.Device(devices[0])
    ty_device.OpenDevice()

    gn_device = mvnc.Device(devices[1])
    gn_device.OpenDevice()

    gn_proc = googlenet_processor(GOOGLENET_GRAPH_FILE, gn_device, gn_input_queue, gn_output_queue,
                                  QUEUE_WAIT_MAX, QUEUE_WAIT_MAX)


    print('Starting GUI, press Q to quit')

    cv2.namedWindow(cv_window_name)
    cv2.waitKey(1)


    frame_count = 0
    start_time = time.time()
    end_time = start_time

    # Queue of camera images.  Only need two spots
    camera_queue = queue.Queue(CAMERA_QUEUE_SIZE)

    # camera processor that will put camera images on the camera_queue
    camera_proc = camera_processor(camera_queue, CAMERA_QUEUE_PUT_WAIT_MAX, CAMERA_INDEX,
                                   CAMERA_REQUEST_VID_WIDTH, CAMERA_REQUEST_VID_HEIGHT,
                                   CAMERA_QUEUE_FULL_SLEEP_SECONDS)
    actual_camera_width = camera_proc.get_actual_camera_width()
    actual_camera_height = camera_proc.get_actual_camera_height()

    ty_output_queue = queue.Queue(TY_OUTPUT_QUEUE_SIZE)
    ty_proc = tiny_yolo_processor(TINY_YOLO_GRAPH_FILE, ty_device, camera_queue, ty_output_queue,
                                  TY_INITIAL_BOX_PROBABILITY_THRESHOLD, TY_INITIAL_MAX_IOU,
                                  QUEUE_WAIT_MAX, QUEUE_WAIT_MAX)

    gn_proc.start_processing()
    camera_proc.start_processing()
    ty_proc.start_processing()

    while True :

        (display_image, filtered_objs) = ty_output_queue.get(True, QUEUE_WAIT_MAX)

        get_googlenet_classifications(display_image, filtered_objs)
        #get_googlenet_classifications_no_queue(gn_proc, display_image, filtered_objs)

        # check if the window is visible, this means the user hasn't closed
        # the window via the X button
        prop_val = cv2.getWindowProperty(cv_window_name, cv2.WND_PROP_ASPECT_RATIO)
        if (prop_val < 0.0):
            end_time = time.time()
            ty_output_queue.task_done()
            break

        overlay_on_image(display_image, filtered_objs)

        # update the GUI window with new image
        cv2.imshow(cv_window_name, display_image)

        raw_key = cv2.waitKey(1)
        if (raw_key != -1):
            if (handle_keys(raw_key) == False):
                end_time = time.time()
                ty_output_queue.task_done()
                break

        frame_count = frame_count + 1

        ty_output_queue.task_done()

    frames_per_second = frame_count / (end_time - start_time)
    print ('Frames per Second: ' + str(frames_per_second))



    camera_proc.stop_processing()
    ty_proc.stop_processing()
    gn_proc.stop_processing()

    camera_proc.cleanup()

    # clean up tiny yolo
    ty_proc.cleanup()
    ty_device.CloseDevice()

    # Clean up googlenet
    gn_proc.cleanup()
    gn_device.CloseDevice()

    print('Finished')


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())
