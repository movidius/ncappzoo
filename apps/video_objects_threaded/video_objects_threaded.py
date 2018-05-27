#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.


from mvnc import mvncapi as mvnc
from video_processor import video_processor
from ssd_mobilenet_processor import ssd_mobilenet_processor
import numpy
import cv2
import time
import os
import queue
import sys
from sys import argv

# labels AKA classes.  The class IDs returned
# are the indices into this list
labels = ('background',
          'aeroplane', 'bicycle', 'bird', 'boat',
          'bottle', 'bus', 'car', 'cat', 'chair',
          'cow', 'diningtable', 'dog', 'horse',
          'motorbike', 'person', 'pottedplant',
          'sheep', 'sofa', 'train', 'tvmonitor')

# only accept classifications with 1 in the class id index.
# default is to accept all object clasifications.
# for example if object_classifications_mask[1] == 0 then
#    will ignore aeroplanes
object_classifications_mask = [1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1]

SSDMN_GRAPH_FILENAME = "./graph"

# the ssd mobilenet image width and height
NETWORK_IMAGE_WIDTH = 300
NETWORK_IMAGE_HEIGHT = 300

# the minimal score for a box to be shown
DEFAULT_INIT_MIN_SCORE = 60
min_score_percent = DEFAULT_INIT_MIN_SCORE

VIDEO_QUEUE_PUT_WAIT_MAX = 4
VIDEO_QUEUE_FULL_SLEEP_SECONDS = 0.01
# for title bar of GUI window
cv_window_name = 'video_objects with SSD_MobileNet'

VIDEO_QUEUE_SIZE = 10
OBJ_DETECT_OUTPUT_QUEUE_SIZE = 10

# number of seconds to wait when putting or getting from queue's
# besides the video output queue.
QUEUE_WAIT_MAX = 2

# the ssd_mobilenet_processor
obj_detector_proc = None

video_proc = None
video_queue = None

# read video files from this directory
input_video_path = '.'

# the resize_window arg will modify these if its specified on the commandline
resize_output = False
resize_output_width = 0
resize_output_height = 0


# create a preprocessed image from the source image that complies to the
# network expectations and return it
def preprocess_image(source_image):
    resized_image = cv2.resize(source_image, (NETWORK_IMAGE_WIDTH, NETWORK_IMAGE_HEIGHT))
    
    # trasnform values from range 0-255 to range -1.0 - 1.0
    resized_image = resized_image - 127.5
    resized_image = resized_image * 0.007843
    return resized_image

# handles key presses by adjusting global thresholds etc.
# raw_key is the return value from cv2.waitkey
# returns False if program should end, or True if should continue
def handle_keys(raw_key):
    global min_score_percent
    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
        return False
    elif (ascii_code == ord('B')):
        min_score_percent += 5
        print('New minimum box percentage: ' + str(min_score_percent) + '%')
    elif (ascii_code == ord('b')):
        min_score_percent -= 5
        print('New minimum box percentage: ' + str(min_score_percent) + '%')

    return True


# overlays the boxes and labels onto the display image.
# display_image is the image on which to overlay the boxes/labels
# object_info is a list of lists which have 6 values each
# these are the 6 values:
#    [0] string that is network classification ie 'cat', or 'chair' etc
#    [1] float value for box X1
#    [2] float value for box Y1
#    [3] float value for box X2
#    [4] float value for box Y2
#    [5] float value that is the probability for the network classification.
# returns None
def overlay_on_image(display_image, object_info_list):
    source_image_width = display_image.shape[1]
    source_image_height = display_image.shape[0]

    for one_object in object_info_list:
        percentage = int(one_object[5] * 100)
        if (percentage <= min_score_percent):
            continue

        label_text = one_object[0] + " (" + str(percentage) + "%)"
        box_left =  int(one_object[1])  # int(object_info[base_index + 3] * source_image_width)
        box_top = int(one_object[2]) # int(object_info[base_index + 4] * source_image_height)
        box_right = int(one_object[3]) # int(object_info[base_index + 5] * source_image_width)
        box_bottom = int(one_object[4])# int(object_info[base_index + 6] * source_image_height)

        box_color = (255, 128, 0)  # box color
        box_thickness = 2
        cv2.rectangle(display_image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

        scale_max = (100.0 - min_score_percent)
        scaled_prob = (percentage - min_score_percent)
        scale = scaled_prob / scale_max

        # draw the classification label string just above and to the left of the rectangle
        label_background_color = (0, int(scale * 175), 75)
        label_text_color = (255, 255, 255)  # white text

        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_left = box_left
        label_top = box_top - label_size[1]
        if (label_top < 1):
            label_top = 1
        label_right = label_left + label_size[0]
        label_bottom = label_top + label_size[1]
        cv2.rectangle(display_image, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1),
                      label_background_color, -1)

        # label text above the box
        cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)


#return False if found invalid args or True if processed ok
def handle_args():
    global resize_output, resize_output_width, resize_output_height, min_score_percent, object_classifications_mask
    for an_arg in argv:
        if (an_arg == argv[0]):
            continue

        elif (str(an_arg).lower() == 'help'):
            return False

        elif (str(an_arg).lower().startswith('exclude_classes=')):
            try:
                arg, val = str(an_arg).split('=', 1)
                exclude_list = str(val).split(',')
                for exclude_id_str in exclude_list:
                    exclude_id = int(exclude_id_str)
                    if (exclude_id < 0 or exclude_id>len(labels)):
                        print("invalid exclude_classes= parameter")
                        return False
                    print("Excluding class ID " + str(exclude_id) + " : " + labels[exclude_id])
                    object_classifications_mask[int(exclude_id)] = 0
            except:
                print('Error with exclude_classes argument. ')
                return False;

        elif (str(an_arg).lower().startswith('init_min_score=')):
            try:
                arg, val = str(an_arg).split('=', 1)
                init_min_score_str = val
                init_min_score = int(init_min_score_str)
                if (init_min_score < 0 or init_min_score > 100):
                    print('Error with init_min_score argument.  It must be between 0-100')
                    return False
                min_score_percent = init_min_score
                print ('Initial Minimum Score: ' + str(min_score_percent) + ' %')
            except:
                print('Error with init_min_score argument.  It must be between 0-100')
                return False;

        elif (str(an_arg).lower().startswith('resize_window=')):
            try:
                arg, val = str(an_arg).split('=', 1)
                width_height = str(val).split('x', 1)
                resize_output_width = int(width_height[0])
                resize_output_height = int(width_height[1])
                resize_output = True
                print ('GUI window resize now on: \n  width = ' +
                       str(resize_output_width) +
                       '\n  height = ' + str(resize_output_height))
            except:
                print('Error with resize_window argument: "' + an_arg + '"')
                return False
        else:
            return False

    return True


# Run an inference on the passed image
# image_to_classify is the image on which an inference will be performed
#    upon successful return this image will be overlayed with boxes
#    and labels identifying the found objects within the image.
# ssd_mobilenet_graph is the Graph object from the NCAPI which will
#    be used to peform the inference.
def run_inference(image_to_classify, ssd_mobilenet_graph):

    # preprocess the image to meet nework expectations
    resized_image = preprocess_image(image_to_classify)

    # Send the image to the NCS as 16 bit floats
    ssd_mobilenet_graph.LoadTensor(resized_image.astype(numpy.float16), None)

    # Get the result from the NCS
    output, userobj = ssd_mobilenet_graph.GetResult()

    #   a.	First fp16 value holds the number of valid detections = num_valid.
    #   b.	The next 6 values are unused.
    #   c.	The next (7 * num_valid) values contain the valid detections data
    #       Each group of 7 values will describe an object/box These 7 values in order.
    #       The values are:
    #         0: image_id (always 0)
    #         1: class_id (this is an index into labels)
    #         2: score (this is the probability for the class)
    #         3: box left location within image as number between 0.0 and 1.0
    #         4: box top location within image as number between 0.0 and 1.0
    #         5: box right location within image as number between 0.0 and 1.0
    #         6: box bottom location within image as number between 0.0 and 1.0

    # number of boxes returned
    num_valid_boxes = int(output[0])

    for box_index in range(num_valid_boxes):
            base_index = 7+ box_index * 7
            if (not numpy.isfinite(output[base_index]) or
                    not numpy.isfinite(output[base_index + 1]) or
                    not numpy.isfinite(output[base_index + 2]) or
                    not numpy.isfinite(output[base_index + 3]) or
                    not numpy.isfinite(output[base_index + 4]) or
                    not numpy.isfinite(output[base_index + 5]) or
                    not numpy.isfinite(output[base_index + 6])):
                # boxes with non finite (inf, nan, etc) numbers must be ignored
                continue

            x1 = max(int(output[base_index + 3] * image_to_classify.shape[0]), 0)
            y1 = max(int(output[base_index + 4] * image_to_classify.shape[1]), 0)
            x2 = min(int(output[base_index + 5] * image_to_classify.shape[0]), image_to_classify.shape[0]-1)
            y2 = min((output[base_index + 6] * image_to_classify.shape[1]), image_to_classify.shape[1]-1)

            # overlay boxes and labels on to the image
            overlay_on_image(image_to_classify, output[base_index:base_index + 7])

    # display text to let user know how to quit
    cv2.rectangle(image_to_classify,(0, 0),(100, 15), (128, 128, 128), -1)
    cv2.putText(image_to_classify, "Q to Quit", (10, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)


# prints usage information
def print_usage():
    print('\nusage: ')
    print('python3 run_video.py [help][resize_window=<width>x<height>]')
    print('')
    print('options:')
    print('  help - prints this message')
    print('  resize_window - resizes the GUI window to specified dimensions')
    print('                  must be formated similar to resize_window=1280x720')
    print('                  Default isto not resize, use size of video frames.')
    print('  init_min_score - set the minimum score for a box to be recognized')
    print('                  must be a number between 0 and 100 inclusive.')
    print('                  Default is: ' + str(DEFAULT_INIT_MIN_SCORE))

    print('  exclude - comma separated list of object class IDs to exclude from following:')
    index = 0
    for oneLabel in labels:
        print("                 class ID " + str(index) + ": " + oneLabel)
        index += 1
    print('            must be a number between 0 and ' + str(len(labels)) + ' inclusive.')
    print('            Default is to exclude none.')

    print('')
    print('Example: ')
    print('python3 run_video.py resize_window=1920x1080 init_min_score=50 exclude_classes=5,11')


# This function is called from the entry point to do
# all the work.
def main():
    global resize_output, resize_output_width, resize_output_height, \
           obj_detector_proc, resize_output, resize_output_width, resize_output_height, video_proc, video_queue

    if (not handle_args()):
        print_usage()
        return 1

    # get list of all the .mp4 files in the image directory
    input_video_filename_list = os.listdir(input_video_path)
    input_video_filename_list = [i for i in input_video_filename_list if i.endswith('.mp4')]
    if (len(input_video_filename_list) < 1):
        # no images to show
        print('No video (.mp4) files found')
        return 1

    # Set logging level to only log errors
    mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, 3)

    devices = mvnc.enumerate_devices()
    if len(devices) < 1:
        print('No NCS device detected.')
        print('Insert device and try again!')
        return 1

    # Pick the first stick to run the network
    # use the first NCS device for ssd mobilenet inferences
    dev_count = 0
    for one_device in devices:
        try:
            ssdmn_device = mvnc.Device(one_device)
            ssdmn_device.open()
            print("opened device " + str(dev_count))
            break;
        except:
            print("Could not open device " + str(dev_count) + ", trying next device")
            pass
        dev_count += 1

    cv2.namedWindow(cv_window_name)
    cv2.moveWindow(cv_window_name, 10,  10)
    cv2.waitKey(1)

    # Queue of video frame images which will be the output for the video processor
    video_queue = queue.Queue(VIDEO_QUEUE_SIZE)

    # Setup ssd_mobilenet_processor that reads from the video queue
    # and writes to its own ty_output_queue
    obj_detector_output_queue = queue.Queue(OBJ_DETECT_OUTPUT_QUEUE_SIZE)
    obj_detector_proc = ssd_mobilenet_processor(SSDMN_GRAPH_FILENAME, ssdmn_device, video_queue, obj_detector_output_queue,
                                                min_score_percent, QUEUE_WAIT_MAX, QUEUE_WAIT_MAX)



    exit_app = False
    while (True):
        for input_video_file in input_video_filename_list :

            # clear all the queues for the cases where this isn't the
            # first video in the list.
            video_queue.queue.clear()
            obj_detector_output_queue.queue.clear()

            # video processor that will put video frames images on the video_queue
            video_proc = video_processor(video_queue,
                                        input_video_path + '/' + input_video_file,
                                        VIDEO_QUEUE_PUT_WAIT_MAX,
                                        VIDEO_QUEUE_FULL_SLEEP_SECONDS)
            video_proc.start_processing()
            obj_detector_proc.start_processing()

            frame_count = 0
            start_time = time.time()
            end_time = start_time

            while(True):
                try:
                    (display_image, filtered_objs) = obj_detector_output_queue.get(True, QUEUE_WAIT_MAX)
                except :
                    print("exception caught in main")
                    raise


                # check if the window is visible, this means the user hasn't closed
                # the window via the X button
                prop_val = cv2.getWindowProperty(cv_window_name, cv2.WND_PROP_ASPECT_RATIO)
                if (prop_val < 0.0):
                    end_time = time.time()
                    exit_app = True
                    break

                overlay_on_image(display_image, filtered_objs)

                if (resize_output):
                    display_image = cv2.resize(display_image,
                                               (resize_output_width, resize_output_height),
                                               cv2.INTER_LINEAR)
                cv2.imshow(cv_window_name, display_image)

                obj_detector_output_queue.task_done()

                raw_key = cv2.waitKey(1)
                if (raw_key != -1):
                    if (handle_keys(raw_key) == False):
                        end_time = time.time()
                        exit_app = True
                        break
                frame_count += 1

                if (video_queue.empty()):
                    end_time = time.time()
                    print('video queue empty')
                    break

            frames_per_second = frame_count / (end_time - start_time)
            print('Frames per Second: ' + str(frames_per_second))

            throttling = ssdmn_device.get_option(mvnc.DeviceOption.RO_THERMAL_THROTTLING_LEVEL)
            if (throttling > 0):
                print("\nDevice is throttling, level is: " + str(throttling) + "\n")

            video_proc.stop_processing()
            video_proc.cleanup()
            cv2.waitKey(1)
            obj_detector_proc.stop_processing()
            cv2.waitKey(1)

            if (exit_app):
                break
        if (exit_app):
            break


    # Clean up the graph and the device
    obj_detector_proc.cleanup()
    ssdmn_device.close()
    ssdmn_device.destroy()

    cv2.destroyAllWindows()


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())
