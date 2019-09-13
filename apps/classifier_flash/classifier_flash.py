#! /usr/bin/env python3

# Copyright(c) 2017-2018 Intel Corporation.
# License: MIT See LICENSE file in root directory.

GREEN = '\033[1;32m'
RED = '\033[1;31m'
NOCOLOR = '\033[0m'
YELLOW = '\033[1;33m'

try:
    from openvino.inference_engine import IENetwork, ExecutableNetwork, IECore
    import openvino.inference_engine.ie_api
except:
    print(RED + '\nPlease make sure your OpenVINO environment variables are set by sourcing the' + YELLOW + ' setupvars.sh ' + RED + 'script found in <your OpenVINO install location>/bin/ folder.\n' + NOCOLOR)
    exit(1)

import cv2
import numpy
import time
import sys
import threading
import os
from sys import argv
import datetime
import queue

from queue import *

INFERENCE_DEV = "MYRIAD"

sep = os.path.sep

#Default vaules
DEFAULT_IMAGE_DIR = "." + sep + "images"
DEFAULT_MODEL_XML = "." + sep + "googlenet-v1.xml"
DEFAULT_MODEL_BIN =  "." + sep + "googlenet-v1.bin"

cv_window_name = "classifier flash"

# how long to wait for queues
inference_device = INFERENCE_DEV
QUEUE_WAIT_SECONDS = 10
DEFAULT_SHOW_FPS = True
DEFAULT_USE_INTERVAL_FPS = False
DEFAULT_DISPLAY_IMAGE_WIDTH = 448
DEFAULT_DISPLAY_IMAGE_HEIGHT = 448
DEFAULT_FRAMES_TO_SKIP = 1

#set some global parameters to initial values that may get overriden with arguments to the appliation.
image_dir = DEFAULT_IMAGE_DIR
number_of_devices = 1
number_of_inferences = 500
run_async = True
time_threads = True
time_main = False

threads_per_dev = 3 # for each device one executable network will be created and this many threads will be

# created to run inferences in parallel on that executable network
# Each thread will start this many async inferences at at time.
# It should be at least the number of NCEs on board.  The Myriad X has 2

simultaneous_infer_per_thread = 6  
report_interval = number_of_inferences / 10 # report out the current FPS every this many inferences
show_fps = DEFAULT_SHOW_FPS
show_device_count = False

display_image_width = DEFAULT_DISPLAY_IMAGE_WIDTH
display_image_height = DEFAULT_DISPLAY_IMAGE_HEIGHT

text_scale = 1.0

model_xml_fullpath = DEFAULT_MODEL_XML
model_bin_fullpath = DEFAULT_MODEL_BIN

use_interval_fps = DEFAULT_USE_INTERVAL_FPS
frames_to_skip = DEFAULT_FRAMES_TO_SKIP

quit_flag = False
pause_flag = False

net_config = {'HW_STAGES_OPTIMIZATION': 'YES', 'COMPUTE_LAYOUT':'VPU_NCHW', 'RESHAPE_OPTIMIZATION':'NO'}

classes_filename = ".." + sep + ".." + sep + 'data/ilsvrc12/synset_words.txt'

INFER_RES_QUEUE_SIZE = 6

def handle_args():
    """Reads the commandline args and adjusts initial values of globals values to match

    :return: False if there was an error with the args, or True if args processed ok.
    """
    global number_of_devices, number_of_inferences, model_xml_fullpath, model_bin_fullpath, run_async, \
           time_threads, time_main, num_ncs_devs, threads_per_dev, simultaneous_infer_per_thread, report_interval, \
           image_dir, show_fps, display_image_width, display_image_height, use_interval_fps, frames_to_skip, inference_device

    have_model_xml = False
    have_model_bin = False

    for an_arg in argv:
        lower_arg = str(an_arg).lower()
        if (an_arg == argv[0]):
            continue

        elif (lower_arg == 'help'):
            return False

        elif (lower_arg.startswith('num_devices=') or lower_arg.startswith("nd=")):
            try:
                arg, val = str(an_arg).split('=', 1)
                num_dev_str = val
                number_of_devices = int(num_dev_str)
                if (number_of_devices < 0):
                    print('Error - num_devices argument invalid.  It must be > 0')
                    return False
                print('setting num_devices: ' + str(number_of_devices))
            except:
                print('Error - num_devices argument invalid.  It must be between 1 and number of devices in system')
                return False;
                
        elif (lower_arg.startswith('device=') or lower_arg.startswith("dev=")):
            try:
                arg, val = str(an_arg).split('=', 1)
                dev = val
                inference_device = str(dev)
                print("inference device:", inference_device)
                if (inference_device != "MYRIAD" and inference_device != "CPU" ):
                    print('Error - Device must be CPU or MYRIAD')
                    return False
                print('setting device: ' + str(inference_device))
            except:
                print('Error - Device must be CPU or MYRIAD')
                return False;
                
        elif (lower_arg.startswith('report_interval=') or lower_arg.startswith("ri=")):
            try:
                arg, val = str(an_arg).split('=', 1)
                val_str = val
                report_interval = int(val_str)
                if (report_interval < 0):
                    print('Error - report_interval must be greater than or equal to 0')
                    return False
                print('setting report_interval: ' + str(report_interval))
            except:
                print('Error - report_interval argument invalid.  It must be greater than or equal to zero')
                return False;

        elif (lower_arg.startswith('frames_to_skip=') or lower_arg.startswith("fts=")):
            try:
                arg, val = str(an_arg).split('=', 1)
                val_str = val
                frames_to_skip = int(val_str)
                if (frames_to_skip < 0):
                    print('Error - frames_to_skip must be greater than or equal to 0')
                    return False
                print('setting frames_to_skip: ' + str(frames_to_skip))
            except:
                print('Error - frames_to_skip argument invalid.  It must be greater than or equal to zero')
                return False;

        elif (lower_arg.startswith('num_inferences=') or lower_arg.startswith('ni=')):
            try:
                arg, val = str(an_arg).split('=', 1)
                num_infer_str = val
                number_of_inferences = int(num_infer_str)
                if (number_of_inferences < 0):
                    print('Error - num_inferences argument invalid.  It must be > 0')
                    return False
                print('setting num_inferences: ' + str(number_of_inferences))
            except:
                print('Error - num_inferences argument invalid.  It must be between 1 and number of devices in system')
                return False;

        elif (lower_arg.startswith('num_threads_per_device=') or lower_arg.startswith('ntpd=')):
            try:
                arg, val = str(an_arg).split('=', 1)
                val_str = val
                threads_per_dev = int(val_str)
                if (threads_per_dev < 0):
                    print('Error - threads_per_dev argument invalid.  It must be > 0')
                    return False
                print('setting num_threads_per_device: ' + str(threads_per_dev))
            except:
                print('Error - num_threads_per_device argument invalid, it must be a positive integer.')
                return False;

        elif (lower_arg.startswith('num_simultaneous_inferences_per_thread=') or lower_arg.startswith('nsipt=')):
            try:
                arg, val = str(an_arg).split('=', 1)
                val_str = val
                simultaneous_infer_per_thread = int(val_str)
                if (simultaneous_infer_per_thread < 0):
                    print('Error - simultaneous_infer_per_thread argument invalid.  It must be > 0')
                    return False
                print('setting num_simultaneous_inferences_per_thread: ' + str(simultaneous_infer_per_thread))
            except:
                print('Error - num_simultaneous_inferences_per_thread argument invalid, it must be a positive integer.')
                return False;

        elif (lower_arg.startswith('model_xml=') or lower_arg.startswith('mx=')):
            try:
                arg, val = str(an_arg).split('=', 1)
                model_xml_fullpath = val
                if not (os.path.isfile(model_xml_fullpath)):
                    print("Error - Model XML file passed does not exist or isn't a file")
                    return False
                print('setting model_xml: ' + str(model_xml_fullpath))
                have_model_xml = True
            except:
                print('Error with model_xml argument.  It must be a valid model file generated by the OpenVINO Model Optimizer')
                return False;

        elif (lower_arg.startswith('model_bin=') or lower_arg.startswith('mb=')):
            try:
                arg, val = str(an_arg).split('=', 1)
                model_bin_fullpath = val
                if not (os.path.isfile(model_bin_fullpath)):
                    print("Error - Model bin file passed does not exist or isn't a file")
                    return False
                print('setting model_bin: ' + str(model_bin_fullpath))
                have_model_bin = True
            except:
                print('Error with model_bin argument.  It must be a valid model file generated by the OpenVINO Model Optimizer')
                return False;

        elif (lower_arg.startswith('run_async=') or lower_arg.startswith('ra=')) :
            try:
                arg, val = str(an_arg).split('=', 1)
                run_async = (val.lower() == 'true')
                print ('setting run_async: ' + str(run_async))
            except:
                print("Error with run_async argument.  It must be 'True' or 'False' ")
                return False;

        elif (lower_arg.startswith('use_interval_fps=') or lower_arg.startswith('uifps=')) :
            try:
                arg, val = str(an_arg).split('=', 1)
                use_interval_fps = (val.lower() == 'true')
                print ('setting use_interval_fps: ' + str(use_interval_fps))
            except:
                print("Error with use_interval_fps argument.  It must be 'True' or 'False' ")
                return False;

        elif (lower_arg.startswith('image_dir=') or lower_arg.startswith('id=')):
            try:
                arg, val = str(an_arg).split('=', 1)
                image_dir = val
                if not (os.path.isdir(image_dir)):
                    print("Error - Image directory passed does not exist or isn't a directory:")
                    print("        passed value: " + image_dir)
                    return False
                print('setting image_dir: ' + str(image_dir))

            except:
                print('Error with model_xml argument.  It must be a valid model file generated by the OpenVINO Model Optimizer')
                return False;

        elif (lower_arg.startswith('display_image_size=') or lower_arg.startswith("dis=")):
            try:
                arg, val = str(an_arg).split('=', 1)
                width_height = str(val).split('x', 1)
                display_image_width = int(width_height[0])
                display_image_height = int(width_height[1])
                print ('setting display_image_width = ' + str(display_image_width) +
                       '\n  display_image_height = ' + str(display_image_height))
            except:
                print('Error with display_image_size argument: "' + an_arg + '"')
                return False

        elif (lower_arg.startswith('time_threads=') or lower_arg.startswith('tt=')) :
            try:
                arg, val = str(an_arg).split('=', 1)
                time_threads = (val.lower() == 'true')
                print ('setting time_threads: ' + str(time_threads))
            except:
                print("Error with time_threads argument.  It must be 'True' or 'False' ")
                return False;

        elif (lower_arg.startswith('time_main=') or lower_arg.startswith('tm=')) :
            try:
                arg, val = str(an_arg).split('=', 1)
                time_main = (val.lower() == 'true')
                print ('setting time_main: ' + str(time_main))
            except:
                print("Error with time_main argument.  It must be 'True' or 'False' ")
                return False;

        elif (lower_arg.startswith('show_fps=') or lower_arg.startswith('sfps=')) :
            try:
                arg, val = str(an_arg).split('=', 1)
                show_fps = (val.lower() == 'true')
                print ('setting show_fps: ' + str(show_fps))
            except:
                print("Error with show_fps argument.  It must be 'True' or 'False' ")
                return False;


    if (time_main == False and time_threads == False):
        print("Error - Both time_threads and time_main args were set to false.  One of these must be true. ")
        return False

    if ((have_model_bin and not have_model_xml) or (have_model_xml and not have_model_bin)):
        print("Error - only one of model_bin and model_xml were specified.  You must specify both or neither.")
        return False

    if (run_async == False) and (simultaneous_infer_per_thread != 1):
        print("Warning - If run_async is False then num_simultaneous_inferences_per_thread must be 1.")
        print("Setting num_simultaneous_inferences_per_thread to 1")
        simultaneous_infer_per_thread = 1

    return True


def print_arg_vals():

    print("")
    print("--------------------------------------------------------")
    print("Current date and time: " + str(datetime.datetime.now()))
    print("")
    print("program arguments:")
    print("------------------")
    print('num_devices: ' + str(number_of_devices))
    print('num_inferences: ' + str(number_of_inferences))
    print('num_threads_per_device: ' + str(threads_per_dev))
    print('num_simultaneous_inferences_per_thread: ' + str(simultaneous_infer_per_thread))
    print('report_interval: ' + str(report_interval))
    print('model_xml: ' + str(model_xml_fullpath))
    print('model_bin: ' + str(model_bin_fullpath))
    print('image_dir: ' + str(image_dir))
    print('run_async: ' + str(run_async))
    print('time_threads: ' + str(time_threads))
    print('time_main: ' + str(time_main))

    print('show_fps: ' + str(show_fps))
    print('use_interval_fps: ' + str(use_interval_fps))
    print('display_image_size = ' + str(display_image_width) + "x" + str(display_image_height))
    print('frames_to_skip = ' + str(frames_to_skip))
    print("--------------------------------------------------------")


def print_usage():
    print('\nusage: ')
    print('python3 classifier_flash [help][num_devices=<number of devices to use>] [num_inference=<number of inferences per device>]')
    print('')
    print('options:')
    print("  num_devices or nd - The number of devices to use for inferencing  ")
    print("                      The value must be between 1 and the total number of devices in the system.")
    print("                      Default is to use 1 device. ")
    print("  num_inferences or ni - The number of inferences to run on each device. ")
    print("                         Default is to run 200 inferences. ")
    print("  report_interval or ri - Report the current FPS every time this many inferences are complete. To surpress reporting set to 0")
    print("                         Default is to report FPS ever 400 inferences. ")
    print("  num_threads_per_device or ntpd - The number of threads to create that will run inferences in parallel for each device. ")
    print("                                   Default is to create 2 threads per device. ")
    print("  num_simultaneous_inferences_per_thread or nsipt - The number of inferences that each thread will create asynchronously. ")
    print("                                                    This should be at least equal to the number of NCEs on board or more.")
    print("                                                    Default is 4 simultaneous inference per thread.")
    print("  model_xml or mx - Full path to the model xml file generated by the model optimizer. ")
    print("                    Default is " + DEFAULT_MODEL_XML)
    print("  model_bin or mb - Full path to the model bin file generated by the model optimizer. ")
    print("                    Default is " + DEFAULT_MODEL_BIN)
    print("  image_dir or id - Path to directory with images to use. ")
    print("                    Default is " + DEFAULT_IMAGE_DIR)
    print("  run_async or ra - Set to true to run asynchronous inferences using two threads per device")
    print("                    Default is True ")
    print("  time_main or tm - Set to true to use the time and calculate FPS from the main loop")
    print("                    Default is False ")
    print("  time_threads or tt - Set to true to use the time and calculate FPS from the time reported from inference threads")
    print("                       Default is True ")
    print("  show_fps or sfps - Set to true to show the FPS in GUI")
    print("                     Default is " + str(DEFAULT_SHOW_FPS))
    print("  use_interval_fps or uifps - Set to true to show the FPS for last interval, or False so show accumlative fps when show_fps is True")
    print("                              Default is " + str(DEFAULT_USE_INTERVAL_FPS))
    print("  display_image_size or dis - Set to size the images should be in the GUI in pixels. Use format WIDTHxHEIGHT")
    print("                              for example dis=1280x720")
    print("                     Default is " + str(DEFAULT_DISPLAY_IMAGE_WIDTH) + "x" + str(DEFAULT_DISPLAY_IMAGE_HEIGHT))
    print("  frames_to_skip or fts - The number of frames to skip when there are frames available to render")
    print("                      The value must be greater than 0. ")
    print("                      Default is to skip " + str(DEFAULT_FRAMES_TO_SKIP) + " frames. ")


def preprocess_image(n:int, c:int, h:int, w:int, image_filename:str) :
    image1 = cv2.imread(image_filename)
    image1 = cv2.resize(image1, (w, h))
    image1 = image1.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    image1 = image1.reshape((n, c, h, w))
    return image1
    

def read_labels(classes_filename:str) :
    labels_list = numpy.loadtxt(classes_filename, str, delimiter='\t')
    for label_index in range(0, len(labels_list)):
        temp = labels_list[label_index].split(',')[0].split(' ', 1)[1]
        labels_list[label_index] = temp
    return labels_list


def handle_keys(raw_key:int):
    """Handles key presses by adjusting global thresholds etc.
    :param raw_key: is the return value from cv2.waitkey
    :return: False if program should end, or True if should continue
    """
    global show_fps, pause_flag

    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
        return False

    elif (ascii_code == ord('f')):
        show_fps = not (show_fps)
        print('New value for show_fps: ' + str(show_fps))

    elif (ascii_code == ord('p')):
        pause_flag = not pause_flag
        print('New value for pause_flag: ' + str(pause_flag))

    return True


def put_text_on_image(display_image: numpy.ndarray, text:str, text_left:int, text_top: int, text_min_width:int):
    global display_image_width
    background_color = (0, 175, 75)
    text_color = (255, 255, 255)  # white text
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1)
    text_width = text_size[0][0]
    text_height = text_size[0][1]

    if (text_left == -1):
        display_image_width = display_image.shape[1]
        #display_image_height = display_image.shape[0]
        text_left = display_image_width - text_width

    if (text_width < text_min_width) :
        bg_width = text_min_width
    else :
        bg_width = text_width

    if (text_width > display_image_width):
        text_width = display_image_width
		
    bg_left = text_left
    bg_right = bg_left + bg_width
    bg_top = text_top - 2
    bg_bottom = bg_top + text_height + 12 # add a padding for letters that drop below bottom
    text_bottom = text_top + text_height

    cv2.rectangle(display_image, (bg_left, bg_top), (bg_right, bg_bottom),
                  background_color, -1)
    cv2.putText(display_image, text, (text_left, text_bottom), cv2.FONT_HERSHEY_SIMPLEX, text_scale,
                text_color, 1)


def overlay_on_image(display_image: numpy.ndarray, label_text:str, fps: float):
    """Overlays the boxes and labels onto the display image.
    :param display_image: the image on which to overlay the boxes/labels
    :return: None
    """
    display_image_width = display_image.shape[1]
    display_image_height = display_image.shape[0]
    put_text_on_image(display_image, label_text, 0, display_image_height - 50, display_image_width)

    if (show_fps):
        fps_str = "FPS: %2.1f" % fps
    else:
        fps_str = "           "
    put_text_on_image(display_image, fps_str, -1, display_image_height - 50, 100)



def from_video_object_scalable(display_image: numpy.ndarray, object_info_list: list, fps: float):
    """Overlays the boxes and labels onto the display image.
    :param display_image: the image on which to overlay the boxes/labels
    :param object_info_list: is a list of lists which have 6 values each
           these are the 6 values:
           [0] string that is network classification ie 'cat', or 'chair' etc
           [1] float value for box upper left X
           [2] float value for box upper left Y
           [3] float value for box lower right X
           [4] float value for box lower right Y
           [5] float value that is the probability 0.0 -1.0 for the network classification.
    :return: None
    """
    source_image_width = display_image.shape[1]
    source_image_height = display_image.shape[0]

    for one_object in object_info_list:
        percentage = int(one_object[5] * 100)

        label_text = one_object[0] + " (" + str(percentage) + "%)"
        box_left = int(one_object[1]) 
        box_top = int(one_object[2])  
        box_right = int(one_object[3])  
        box_bottom = int(one_object[4])  

        box_color = (255, 128, 0)  # box color
        box_thickness = 2
        cv2.rectangle(display_image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

        min_score_percent = 80
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
        cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    label_text_color, 1)

    if (show_fps):
        fps_text = "FPS: " + "{:.2f}".format(fps)
        fps_thickness = 2
        fps_multiplier = 1.5
        fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, fps_multiplier, fps_thickness)[0]
        text_pad = 10
        box_coord_left = 0
        box_coord_top = 0
        box_coord_right = box_coord_left + fps_size[0] + text_pad * 2
        box_coord_bottom = box_coord_top + fps_size[1] + text_pad * 2

        fps_left = box_coord_left + text_pad
        fps_bottom = box_coord_bottom - text_pad
        label_background_color = (200, 200, 200)
        label_text_color = (255, 0, 0)

        fps_image = numpy.full((box_coord_bottom - box_coord_top, box_coord_right - box_coord_left, 3),
                               label_background_color, numpy.uint8)
        cv2.putText(fps_image, fps_text, (fps_left, fps_bottom), cv2.FONT_HERSHEY_SIMPLEX, fps_multiplier,
                    label_text_color, fps_thickness)

        fps_transparency = 0.4
        cv2.addWeighted(display_image[box_coord_top:box_coord_bottom, box_coord_left:box_coord_right],
                        1.0 - fps_transparency,
                        fps_image, fps_transparency, 0.0,
                        display_image[box_coord_top:box_coord_bottom, box_coord_left:box_coord_right])

    device_count = 1
    if (show_device_count):
        ncs_count_text = "Devices: " + str(device_count)

        ncs_count_thickness = 2
        ncs_count_multiplier = 1.5
        ncs_count_size = \
        cv2.getTextSize(ncs_count_text, cv2.FONT_HERSHEY_SIMPLEX, ncs_count_multiplier, ncs_count_thickness)[0]
        ncs_count_text_pad = 10
        ncs_count_box_coord_left = display_image.shape[1] - ncs_count_size[0] - + ncs_count_text_pad * 2  # 0
        ncs_count_box_coord_top = 0
        ncs_count_box_coord_right = ncs_count_box_coord_left + ncs_count_size[0] + ncs_count_text_pad * 2
        ncs_count_box_coord_bottom = ncs_count_box_coord_top + ncs_count_size[1] + ncs_count_text_pad * 2

        ncs_count_label_background_color = (200, 200, 200)
        ncs_count_label_text_color = (255, 0, 0)

        ncs_count_image = numpy.full((ncs_count_box_coord_bottom - ncs_count_box_coord_top,
                                      ncs_count_box_coord_right - ncs_count_box_coord_left, 3),
                                     ncs_count_label_background_color, numpy.uint8)
        cv2.putText(ncs_count_image, ncs_count_text, (0 + ncs_count_text_pad, ncs_count_size[1] + ncs_count_text_pad),
                    cv2.FONT_HERSHEY_SIMPLEX, ncs_count_multiplier, ncs_count_label_text_color, ncs_count_thickness)

        ncs_count_transparency = 0.4
        cv2.addWeighted(display_image[ncs_count_box_coord_top:ncs_count_box_coord_bottom,
                        ncs_count_box_coord_left:ncs_count_box_coord_right], 1.0 - ncs_count_transparency,
                        ncs_count_image, ncs_count_transparency, 0.0,
                        display_image[ncs_count_box_coord_top:ncs_count_box_coord_bottom,
                        ncs_count_box_coord_left:ncs_count_box_coord_right])


def main():
    """Main function for the program.  Everything starts here.

    :return: None
    """

    global quit_flag, pause_flag, total_paused_time

    if (handle_args() != True):
        print_usage()
        exit()

    print_arg_vals()
    # Read in the labels and get the number of ncs devices
    labels_list = read_labels(classes_filename)
    num_ncs_devs = number_of_devices
    
    # calculate numberof inferences that need to be made per thread/device
    inferences_per_thread = int(number_of_inferences / ((threads_per_dev * num_ncs_devs)))
    inferences_per_thread = int(inferences_per_thread / simultaneous_infer_per_thread) * simultaneous_infer_per_thread
    total_number_threads = num_ncs_devs * threads_per_dev

    infer_result_queue = queue.Queue(INFER_RES_QUEUE_SIZE)
    # create a thread list and a results time list to track fps
    result_times_list = [None] * (num_ncs_devs * threads_per_dev)
    thread_list = [None] * (num_ncs_devs * threads_per_dev)
    # barrier to sync all threads
    start_barrier = threading.Barrier(num_ncs_devs*threads_per_dev+1)
    end_barrier = threading.Barrier(num_ncs_devs*threads_per_dev+1)

    canvas_width = display_image_width
    canvas_height = display_image_height
    canvas_image = numpy.zeros((canvas_height, canvas_width, 3), numpy.uint8)
    # OpenCV window initialization
    cv2.namedWindow(cv_window_name)
    cv2.moveWindow(cv_window_name, 10,  10)
    cv2.waitKey(1)
    put_text_on_image(canvas_image, "Loading Images...", 0, int(canvas_height/2) - 20, canvas_width)
    cv2.imshow(cv_window_name, canvas_image)
    cv2.waitKey(1)

    total_paused_time = 0.0

    # load a single plugin for the application
    ie = IECore()
    net = IENetwork(model=model_xml_fullpath, weights=model_bin_fullpath)
    input_blob = next(iter(net.inputs))
    output_blob = next(iter(net.outputs))

    n, c, h, w = net.inputs[input_blob].shape

    # get list of all the .mp4 files in the image directory
    image_filename_list = os.listdir(image_dir)
    image_filename_list = [image_dir + sep + i for i in image_filename_list if (i.endswith('.jpg') or i.endswith(".png"))]
    if (len(image_filename_list) < 1):
        # no images to show
        print('No image files found (.jpg or .png)')
        return 1

    print("Found " + str(len(image_filename_list)) + " images.")
    
    # resizing all of the images in the image list
    display_image_list = [None]*len(image_filename_list)
    preprocessed_image_list = [None]*len(image_filename_list)
    preprocessed_image_index = 0
    for one_image_filename in image_filename_list:
        display_image_list[preprocessed_image_index] = cv2.imread(one_image_filename)
        display_image_list[preprocessed_image_index] = cv2.resize(display_image_list[preprocessed_image_index], (display_image_width, display_image_height))
        one_preprocessed_image = preprocess_image(n,c,h,w,one_image_filename)
        preprocessed_image_list[preprocessed_image_index] = one_preprocessed_image
        preprocessed_image_index += 1
        cv2.waitKey(1)

    images_per_thread = int(len(preprocessed_image_list) / total_number_threads)

    exec_net_list = [None] * num_ncs_devs

    # Load the network for each device and
    for dev_index in range(0, num_ncs_devs):
        # create one executable network for each device in the system
        # create 4 requests for each executable network, two for each NCE
        exec_net_list[dev_index] = ie.load_network(network=net, num_requests=threads_per_dev*simultaneous_infer_per_thread, device_name = inference_device)

        # create threads for each executable network (one executable network per device)
        for dev_thread_index in range(0,threads_per_dev):
            total_thread_index = dev_thread_index + (threads_per_dev*dev_index)
            first_image_index = int(total_thread_index*images_per_thread)
            last_image_index = int(first_image_index + images_per_thread - 1)

            print("first image index: " + str(first_image_index) + "  last_image_index: " + str(last_image_index))
            # Threads will run async inference once they start
            if (run_async):
                thread_list[total_thread_index] = threading.Thread(target=infer_async_thread_proc,
                                                                              args=[exec_net_list[dev_index], dev_thread_index*simultaneous_infer_per_thread,
                                                                                    preprocessed_image_list, image_filename_list, display_image_list,
                                                                                    first_image_index, last_image_index,
                                                                                    inferences_per_thread,
                                                                                    result_times_list, total_thread_index,
                                                                                    start_barrier, end_barrier, simultaneous_infer_per_thread,
                                                                                    infer_result_queue, input_blob, output_blob])
            else:
                pass

    del net

    #start the threads
    for one_thread in thread_list:
        one_thread.start()

    start_barrier.wait()

    # save the main starting time
    main_start_time = time.time()
    interval_start_time = time.time()

    print("Inferences started.")

    cur_fps = 0.0
    result_counter = 0
    accum_fps = 0.0
    frames_since_last_report = 0

    # track the inference results and display the images
    while (result_counter < total_number_threads * inferences_per_thread):
        infer_res = infer_result_queue.get(True, QUEUE_WAIT_SECONDS)
        if (infer_result_queue.qsize() > (frames_to_skip)):
            for extra_res_counter in range(0, frames_to_skip):
                infer_result_queue.task_done()
                result_counter += 1
                frames_since_last_report += 1
                infer_res = infer_result_queue.get(True, QUEUE_WAIT_SECONDS)

        infer_res_filename = infer_res[0]
        infer_res_index = infer_res[1]
        infer_res_probability = infer_res[2]
        infer_res_display_image = infer_res[3]

        fps_to_use = accum_fps
        if (use_interval_fps):
            fps_to_use = cur_fps

        overlay_on_image(infer_res_display_image, labels_list[infer_res_index], fps_to_use)
        # Display image in the window
        cv2.imshow(cv_window_name, infer_res_display_image)
        
        # Handle key events
        raw_key = cv2.waitKey(1)
        if (raw_key != -1 or quit_flag):
            if (handle_keys(raw_key) == False or quit_flag):
                print("Quitting application")
                quit_flag = True
                pause_flag = False
                exception_count = 0
                max_exceptions = 50000
                while(end_barrier.n_waiting < total_number_threads) :
                    try:
                        infer_res = infer_result_queue.get_nowait()
                    except:
                        exception_count += 1
                    if (exception_count >= max_exceptions):
                        break
                print("Exception Count: " + str(exception_count))
                if (exception_count >= max_exceptions):
                    print("Gave up waiting for threads to finish, exiting!")
                    cv2.destroyAllWindows()
                    exit(-1)
                break
            if (pause_flag):
                pause_start_time = time.time()
                while(pause_flag and not quit_flag):
                    raw_key = cv2.waitKey(1)
                    if (raw_key != -1):
                        if (handle_keys(raw_key) == False):
                            quit_flag = True
                pause_end_time = time.time()
                total_paused_time += (pause_end_time - pause_start_time)

        # Report the fps results in intervals
        result_counter += 1
        frames_since_last_report += 1
        if (report_interval > 0):
            if ((frames_since_last_report > report_interval)):
                cur_time = time.time()
                accum_duration = cur_time - main_start_time
                cur_duration = cur_time - interval_start_time

                cur_fps = frames_since_last_report / cur_duration
                accum_fps = result_counter / (accum_duration - total_paused_time)
                print(" " + str(result_counter) + " inferences completed. FPS: " + str("{0:.2f}".format(accum_fps)))
                frames_since_last_report = 0
                interval_start_time = time.time()

        infer_result_queue.task_done()

    # wait for all the inference threads to reach end barrier
    print("main end barrier reached")
    end_barrier.wait()

    # save main end time
    main_end_time = time.time()
    print("Inferences finished.")

    # wait for threads to finish
    for one_thread in thread_list:
        one_thread.join()

    total_thread_fps = 0.0
    total_thread_time = 0.0
    # Calculate overall fps
    for thread_index in range(0, (num_ncs_devs*threads_per_dev)):
        total_thread_time += (result_times_list[thread_index] - total_paused_time)
        total_thread_fps += (inferences_per_thread / (result_times_list[thread_index] - total_paused_time))

    if (quit_flag):
        # adjust the thread fps since didn't do every inference.
        total_thread_fps = result_counter / (total_thread_time / total_number_threads)

    devices_count = str(number_of_devices)

    if (time_threads):
        print("\n------------------- Thread timing -----------------------")
        print("--- Device: " + str(inference_device))
        print("--- Model:  " + model_xml_fullpath)
        print("--- Total FPS: " + '{0:.1f}'.format(total_thread_fps))
        print("--- FPS per device: " + '{0:.1f}'.format(total_thread_fps / num_ncs_devs))
        print("---------------------------------------------------------")

    main_time = (main_end_time - main_start_time) - total_paused_time

    if (time_main):
        main_fps = result_counter / main_time
        print ("\n------------------ Main timing -------------------------")
        print ("--- FPS: " + str(main_fps))
        print ("--- FPS per device: " + str(main_fps/num_ncs_devs))
        print ("--------------------------------------------------------")

    # clean up
    for one_exec_net in exec_net_list:
        del one_exec_net


# use this thread proc to try to implement:
#  1 plugin per app
#  1 executable Network per device
#  multiple threads per executable network
#  multiple requests per executable network per thread
def infer_async_thread_proc(exec_net: ExecutableNetwork, first_request_index: int,
                            image_list: list, image_filename_list: list, display_image_list: list,
                            first_image_index:int, last_image_index:int,
                            num_total_inferences: int, result_list: list, result_index:int,
                            start_barrier: threading.Barrier, end_barrier: threading.Barrier,
                            simultaneous_infer_per_thread:int, infer_result_queue:queue.Queue, input_blob, output_blob):

    # image_list is list is of numpy.ndarray (preprocessed images)
    # image_filename_list is list of strings are filename of corresponding image in image_list

    # sync with the main start barrier
    start_barrier.wait()

    start_time = time.time()
    end_time = start_time

    handle_list = [None]*simultaneous_infer_per_thread

    image_index = first_image_index

    image_result_start_index = 0

    # do all work to be done by the thread
    for outer_index in range(0, int(num_total_inferences/simultaneous_infer_per_thread)):

        # Start the simultaneous async inferences
        for start_index in range(0, simultaneous_infer_per_thread):

            # handle_list
            handle_list[start_index] = exec_net.start_async(request_id=first_request_index+start_index, inputs={input_blob: image_list[image_index]}), image_filename_list[image_index], display_image_list[image_index]
            image_index += 1
            if (image_index > last_image_index):
                image_index = first_image_index

        # Wait for the simultaneous async inferences to finish.
        for wait_index in range(0, simultaneous_infer_per_thread):
            res = None
            infer_stat = handle_list[wait_index][0].wait()
            res = handle_list[wait_index][0].outputs[output_blob]
            top_ind = numpy.argsort(res, axis=1)[0, -1:][::-1]
            top_ind = top_ind[0]
            prob = res[0][top_ind]
            image_filename = handle_list[wait_index][1]
            display_image = handle_list[wait_index][2]

            # put a tuple on the output queue with (filename, top index, probability, and display_image)

            infer_result_queue.put((image_filename, top_ind, prob, display_image), True)

            handle_list[wait_index] = None

        if (quit_flag == True):
            # the quit flag was set from main so break out of loop
            break


    # save the time spent on inferences within this inference thread and associated reader thread
    end_time = time.time()
    total_inference_time = end_time - start_time
    result_list[result_index] = total_inference_time

    print("thread " + str(result_index) + " end barrier reached")

    # wait for all inference threads to finish
    end_barrier.wait()


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())
