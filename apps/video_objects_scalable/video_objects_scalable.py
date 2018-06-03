#! /usr/bin/env python3

# Copyright(c) 2017-2018 Intel Corporation.
# License: MIT See LICENSE file in root directory.


from mvnc import mvncapi as mvnc
from video_processor import VideoProcessor
from ssd_mobilenet_processor import SsdMobileNetProcessor
import cv2
import numpy
import time
import os
import sys
from sys import argv

# only accept classifications with 1 in the class id index.
# default is to accept all object clasifications.
# for example if object_classifications_mask[1] == 0 then
#    will ignore aeroplanes
object_classifications_mask = [1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1]

NETWORK_GRAPH_FILENAME = "./graph"

# the minimal score for a box to be shown
DEFAULT_INIT_MIN_SCORE = 60
min_score_percent = DEFAULT_INIT_MIN_SCORE

# for title bar of GUI window
cv_window_name = 'video_objects_threaded - SSD_MobileNet'

# read video files from this directory
input_video_path = '.'

# the resize_window arg will modify these if its specified on the commandlineq
resize_output = False
resize_output_width = 0
resize_output_height = 0

DEFAULT_SHOW_FPS = True
show_fps = DEFAULT_SHOW_FPS


def handle_keys(raw_key:int, obj_detector_list:list):
    """Handles key presses by adjusting global thresholds etc.
    :param raw_key: is the return value from cv2.waitkey
    :param obj_detector_list: list of object detectors the object detector in use.
    :return: False if program should end, or True if should continue
    """
    global min_score_percent, show_fps
    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
        return False
    elif (ascii_code == ord('B')):
        min_score_percent = obj_detector_list[0].get_box_probability_threshold() * 100.0 + 5
        if (min_score_percent > 100.0): min_score_percent = 100.0
        for one_object_detect in obj_detector_list:
            one_object_detect.set_box_probability_threshold(min_score_percent/100.0)
        print('New minimum box percentage: ' + str(min_score_percent) + '%')
    elif (ascii_code == ord('b')):
        min_score_percent = obj_detector_list[0].get_box_probability_threshold() * 100.0 - 5
        if (min_score_percent < 0.0): min_score_percent = 0.0
        for one_object_detect in obj_detector_list:
            one_object_detect.set_box_probability_threshold(min_score_percent/100.0)
        print('New minimum box percentage: ' + str(min_score_percent) + '%')

    elif (ascii_code == ord('f')):
        show_fps = not (show_fps)
        print('New value for show_fps: ' + str(show_fps))

    return True



def overlay_on_image(display_image:numpy.ndarray, object_info_list:list, fps:float):
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
        fps_right = box_coord_right - text_pad
        fps_top = box_coord_top + text_pad
        fps_bottom = box_coord_bottom - text_pad
        label_background_color = (200, 200, 200)
        label_text_color = (255, 0, 0)

        fps_image = numpy.full((box_coord_bottom - box_coord_top, box_coord_right - box_coord_left, 3), label_background_color, numpy.uint8)
        cv2.putText(fps_image, fps_text, (fps_left, fps_bottom), cv2.FONT_HERSHEY_SIMPLEX, fps_multiplier, label_text_color,  fps_thickness)

        fps_transparency = 0.4
        cv2.addWeighted(display_image[box_coord_top:box_coord_bottom, box_coord_left:box_coord_right], 1.0 - fps_transparency,
                        fps_image, fps_transparency, 0.0, display_image[box_coord_top:box_coord_bottom, box_coord_left:box_coord_right])

def handle_args():
    """Reads the commandline args and adjusts initial values of globals values to match

    :return: False if there was an error with the args, or True if args processed ok.
    """
    global resize_output, resize_output_width, resize_output_height, min_score_percent, object_classifications_mask,\
           show_fps

    labels = SsdMobileNetProcessor.get_classification_labels()

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

        elif (str(an_arg).lower().startswith('show_fps=')):
            try:
                arg, val = str(an_arg).split('=', 1)
                show_fps = (val.lower() == 'true')
                print ('show_fps: ' + str(show_fps))
            except:
                print("Error with show_fps argument.  It must be 'True' or 'False' ")
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


def print_usage():
    """Prints usage information for the program.

    :return: None
    """
    labels = SsdMobileNetProcessor.get_classification_labels()

    print('\nusage: ')
    print('python3 run_video.py [help][resize_window=<width>x<height>]')
    print('')
    print('options:')
    print('  help - Prints this message')
    print('  resize_window - Resizes the GUI window to specified dimensions')
    print('                  must be formated similar to resize_window=1280x720')
    print('                  Default isto not resize, use size of video frames.')

    print('  init_min_score - Set the minimum score for a box to be recognized')
    print('                   must be a number between 0 and 100 inclusive.')
    print('                   Default is: ' + str(DEFAULT_INIT_MIN_SCORE))

    print("  show_fps - Show or do not show the Frames Per Second while running")
    print("             must be 'True' or 'False'.")
    print("             Default is: " + str(DEFAULT_SHOW_FPS))

    print('  exclude_classes - Comma separated list of object class IDs to exclude from following:')
    index = 0
    for oneLabel in labels:
        print("                 class ID " + str(index) + ": " + oneLabel)
        index += 1
    print('            must be a number between 0 and ' + str(len(labels)-1) + ' inclusive.')
    print('            Default is to exclude none.')

    print('')
    print('Example: ')
    print('python3 run_video.py resize_window=1920x1080 sinit_min_score=50 show_fps=False exclude_classes=5,11')


def main():
    """Main function for the program.  Everything starts here.

    :return: None
    """
    global resize_output, resize_output_width, resize_output_height, \
           resize_output, resize_output_width, resize_output_height

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

    # Create an object detector processor for each device that opens
    # and store it in our list of processors
    obj_detect_list = list()
    dev_count = 0
    for one_device in devices:
        try:
            obj_detect_dev = mvnc.Device(one_device)
            obj_detect_dev.open()
            print("opened device " + str(dev_count))
            obj_detector_proc = SsdMobileNetProcessor(NETWORK_GRAPH_FILENAME, obj_detect_dev,
                                                      inital_box_prob_thresh=min_score_percent / 100.0,
                                                      classification_mask=object_classifications_mask,
                                                      name="object detector " + str(dev_count))
            obj_detect_list.append(obj_detector_proc)

        except:
            print("Could not open device " + str(dev_count) + ", trying next device")
            pass

        dev_count += 1

    if len(obj_detect_list) < 1:
        print('Could not open any NCS devices.')
        print('Reinsert devices and try again!')
        return 1

    print("Using " + str(len(obj_detect_list)) + " devices for object detection")

    cv2.namedWindow(cv_window_name)
    cv2.moveWindow(cv_window_name, 10,  10)
    cv2.waitKey(1)

    exit_app = False
    while (True):
        for input_video_file in input_video_filename_list :

            for one_obj_detect_proc in obj_detect_list:
                one_obj_detect_proc.drain_queues()

            # video processor that will put video frames images on the object detector's input FIFO queue
            video_proc = VideoProcessor(input_video_path + '/' + input_video_file,
                                        network_processor_list = obj_detect_list)
            video_proc.start_processing()

            frame_count = 0
            start_time = time.time()
            end_time = start_time

            while(True):
                done = False
                for one_obj_detect_proc in obj_detect_list:
                    try:
                        (filtered_objs, display_image) = one_obj_detect_proc.get_async_inference_result()
                    except :
                        print("exception caught in main")
                        raise


                    # check if the window is visible, this means the user hasn't closed
                    # the window via the X button
                    prop_val = cv2.getWindowProperty(cv_window_name, cv2.WND_PROP_ASPECT_RATIO)
                    if (prop_val < 0.0):
                        end_time = time.time()
                        video_proc.stop_processing()
                        video_proc.cleanup()
                        exit_app = True
                        break

                    running_fps = frame_count / (time.time() - start_time)
                    overlay_on_image(display_image, filtered_objs, running_fps)

                    if (resize_output):
                        display_image = cv2.resize(display_image,
                                                   (resize_output_width, resize_output_height),
                                                   cv2.INTER_LINEAR)
                    cv2.imshow(cv_window_name, display_image)

                    raw_key = cv2.waitKey(1)
                    if (raw_key != -1):
                        if (handle_keys(raw_key, obj_detect_list) == False):
                            end_time = time.time()
                            exit_app = True
                            done = True
                            break

                    frame_count += 1

                    #if (one_obj_detect_proc.is_input_queue_empty()):
                    if (not video_proc.is_processing()):
                        # asssume the video is over.
                        end_time = time.time()
                        done = True
                        print('video processor not processing, assuming video is finished.')
                        break

                if (done) : break

            frames_per_second = frame_count / (end_time - start_time)
            print('Frames per Second: ' + str(frames_per_second))

            #throttling = obj_detect_dev.get_option(mvnc.DeviceOption.RO_THERMAL_THROTTLING_LEVEL)
            #if (throttling > 0):
            #    print("\nDevice is throttling, level is: " + str(throttling))
            #    print("Sleeping for a few seconds....")
            #    cv2.waitKey(2000)

            video_proc.stop_processing()
            video_proc.cleanup()

            if (exit_app):
                break

        if (exit_app):
            break


    # Clean up the graph and the device
    for one_obj_detect_proc in obj_detect_list:
        cv2.waitKey(1)
        one_obj_detect_proc.cleanup(True)

    cv2.destroyAllWindows()


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())
