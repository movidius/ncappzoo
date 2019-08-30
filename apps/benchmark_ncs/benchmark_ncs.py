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

DEFAULT_IMAGE_DIR = "." + sep + "images"
DEFAULT_MODEL_XML = "." + sep + "googlenet-v1.xml"
DEFAULT_MODEL_BIN =  "." + sep + "googlenet-v1.bin"

cv_window_name = "benchmark_ncs"

# how long to wait for queues
QUEUE_WAIT_SECONDS = 10

# set some global parameters to initial values that may get overriden with arguments to the application.
inference_device = INFERENCE_DEV
image_dir = DEFAULT_IMAGE_DIR
number_of_devices = 1
number_of_inferences = 1000
run_async = True
time_threads = True
time_main = False

threads_per_dev = 3 # for each device one executable network will be created and this many threads will be

simultaneous_infer_per_thread = 6  # Each thread will start this many async inferences at at time.
                                   # it should be at least the number of NCEs on board.  The Myriad X has 2
                                   # seem to get slightly better results more. Myriad X does well with 4
report_interval = int(number_of_inferences / 10) #report out the current FPS every this many inferences

model_xml_fullpath = DEFAULT_MODEL_XML
model_bin_fullpath = DEFAULT_MODEL_BIN

net_config = {'HW_STAGES_OPTIMIZATION': 'YES', 'COMPUTE_LAYOUT':'VPU_NCHW', 'RESHAPE_OPTIMIZATION':'NO'}


INFER_RES_QUEUE_SIZE = 6

def handle_args():
    """Reads the commandline args and adjusts initial values of globals values to match

    :return: False if there was an error with the args, or True if args processed ok.
    """
    global number_of_devices, number_of_inferences, model_xml_fullpath, model_bin_fullpath, run_async, \
           time_threads, time_main, num_ncs_devs, threads_per_dev, simultaneous_infer_per_thread, report_interval, \
           image_dir, inference_device

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
    print('device: ' + inference_device)
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
    print("--------------------------------------------------------")


def print_usage():
    print('\nusage: ')
    print('python3 benchmark_ncs [help][nd=<number of devices to use>] [ni=<number of inferences per device>]')
    print('                      [report_interval=<num inferences between reporting>] [ntpd=<number of threads to use per device>]')
    print('                      [nsipt=<simultaneous inference on each thread>] [mx=<path to model xml file> mb=<path to model bin file>]')
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

def preprocess_image(n:int, c:int, h:int, w:int, image_filename:str) :
    image = cv2.imread(image_filename)
    image = cv2.resize(image, (w, h))
    image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    preprocessed_image = image.reshape((n, c, h, w))
    return preprocessed_image


def main():
    """Main function for the program.  Everything starts here.

    :return: None
    """

    if (handle_args() != True):
        print_usage()
        exit()

    print_arg_vals()

    num_ncs_devs = number_of_devices

    # Calculate the number of number of inferences to be made per thread
    total_number_of_threads = threads_per_dev * num_ncs_devs   
    inferences_per_thread = int(number_of_inferences / total_number_of_threads)
    # This total will be the total number of inferences to be made
    inferences_per_thread = int(inferences_per_thread / simultaneous_infer_per_thread) * simultaneous_infer_per_thread
    
    # Total number of threads that need to be spawned
    total_number_threads = num_ncs_devs * threads_per_dev
    
    # Lists and queues to hold data
    infer_result_queue = queue.Queue(INFER_RES_QUEUE_SIZE)
    infer_time_list = [None] * (total_number_threads)
    thread_list = [None] * (total_number_threads)
    
    # Threading barrier to sync all thread processing
    start_barrier = threading.Barrier(total_number_threads + 1)
    end_barrier = threading.Barrier(total_number_threads + 1)

    ie = IECore()
    
    # Create the network object 
    net = IENetwork(model=model_xml_fullpath, weights=model_bin_fullpath)
    # Get the input and output blob names and the network input information
    input_blob = next(iter(net.inputs))
    output_blob = next(iter(net.outputs))
    n, c, h, w = net.inputs[input_blob].shape

    # Get a list of all the .mp4 files in the image directory and put them in a list
    image_filename_list = os.listdir(image_dir)
    image_filename_list = [image_dir + sep + i for i in image_filename_list if (i.endswith('.jpg') or i.endswith(".png"))]
    if (len(image_filename_list) < 1):
        # no images to show
        print('No image files found (.jpg or .png)')
        return 1
    print("Found " + str(len(image_filename_list)) + " images.")

    # Preprocess all images in the list
    preprocessed_image_list = [None]*len(image_filename_list)
    preprocessed_image_index = 0
    for one_image_filename in image_filename_list:
        one_preprocessed_image = preprocess_image(n,c,h,w,one_image_filename)
        preprocessed_image_list[preprocessed_image_index] = one_preprocessed_image
        preprocessed_image_index += 1

    # Number of images to be inferred per thread
    images_per_thread = int(len(preprocessed_image_list) / total_number_threads)
    
    exec_net_list = [None] * num_ncs_devs

    # creates an executable network for each device
    for dev_index in range(0, num_ncs_devs):
        exec_net_list[dev_index] = ie.load_network(network = net, num_requests = threads_per_dev * simultaneous_infer_per_thread, device_name=inference_device)
        # create threads (3) for each executable network (one executable network per device)
        for dev_thread_index in range(0, threads_per_dev):
            # divide up the images to be processed by each thread
            # device thread index starts at 0. device indexes start at 0. 3 threads per device. (0, 1, 2)
            total_thread_index = dev_thread_index + (threads_per_dev * dev_index)
            # Find out which index in preprocessed_image_list to start from
            first_image_index = int(total_thread_index * images_per_thread)
            # Find out which index in preprocessed_image_list to stop at
            last_image_index = int(first_image_index + images_per_thread - 1)

            if (run_async):
                dev_thread_req_id = dev_thread_index * simultaneous_infer_per_thread
                thread_list[total_thread_index] = threading.Thread(target=infer_async_thread_proc,
                                                                              args=[net, exec_net_list[dev_index], dev_thread_req_id,
                                                                                    preprocessed_image_list,
                                                                                    first_image_index, last_image_index,
                                                                                    inferences_per_thread,
                                                                                    infer_time_list, total_thread_index,
                                                                                    start_barrier, end_barrier, simultaneous_infer_per_thread,
                                                                                    infer_result_queue, input_blob, output_blob], daemon = True)
            else:
                print("run_async=false not yet supported")
                exit(-1)

    del net

    # Start the threads
    try:
        for one_thread in thread_list:
            one_thread.start()
    except (KeyboardInterrupt, SystemExit):
        cleanup_stop_thread()
        sys.exit()
        
    start_barrier.wait()

    # Save the main starting time
    main_start_time = time.time()
    interval_start_time = time.time()

    print("Inferences started...")

    cur_fps = 0.0
    result_counter = 0
    accum_fps = 0.0
    frames_since_last_report = 0
    total_number_inferences = total_number_threads * inferences_per_thread
    
    # Report intermediate results
    while (result_counter < total_number_inferences):
        # Get the number of completed inferences
        infer_res = infer_result_queue.get(True, QUEUE_WAIT_SECONDS)

        infer_res_index = infer_res[0]
        infer_res_probability = infer_res[1]

        result_counter += 1
            
        frames_since_last_report += 1
        if (report_interval > 0):
            if ((frames_since_last_report > report_interval)):
                cur_time = time.time()
                accum_duration = cur_time - main_start_time
                cur_duration = cur_time - interval_start_time
                cur_fps = frames_since_last_report / cur_duration
                accum_fps = result_counter / accum_duration
                print(str(result_counter) + " inferences completed. Current fps: " + '{0:.1f}'.format(accum_fps))
                frames_since_last_report = 0
                interval_start_time = time.time()

        infer_result_queue.task_done()
    
    # wait for all the inference threads to reach end barrier
    print("Main end barrier reached")
    end_barrier.wait()

    # Save main end time
    main_end_time = time.time()
    print("Inferences finished.")

    # wait for all threads to finish
    for one_thread in thread_list:
        one_thread.join()

    total_thread_fps = 0.0
    total_thread_time = 0.0
    # Calculate total time and fps
    for thread_index in range(0, (num_ncs_devs*threads_per_dev)):
        total_thread_time += infer_time_list[thread_index]
        total_thread_fps += (inferences_per_thread / infer_time_list[thread_index])

    devices_count = str(number_of_devices)


    if (time_threads):
        print("\n------------------- Thread timing -----------------------")
        print("--- Device: " + str(inference_device))
        print("--- Model:  " + model_xml_fullpath)
        print("--- Total FPS: " + '{0:.1f}'.format(total_thread_fps))
        print("--- FPS per device: " + '{0:.1f}'.format(total_thread_fps / num_ncs_devs))
        print("---------------------------------------------------------")


    main_time = main_end_time - main_start_time

    if (time_main):
        main_fps = result_counter / main_time
        print ("\n------------------ Main timing -------------------------")
        print ("--- FPS: " + str(main_fps))
        print ("--- FPS per device: " + str(main_fps/num_ncs_devs))
        print ("--------------------------------------------------------")

    # Clean up
    for one_exec_net in exec_net_list:
        del one_exec_net


# use this thread proc to try to implement:
#  1 plugin per app
#  1 executable Network per device
#  multiple threads per executable network
#  multiple requests per executable network per thread
def infer_async_thread_proc(net, exec_net: ExecutableNetwork, dev_thread_request_id: int,
                            image_list: list,
                            first_image_index:int, last_image_index:int,
                            num_total_inferences: int, result_list: list, result_index:int,
                            start_barrier: threading.Barrier, end_barrier: threading.Barrier,
                            simultaneous_infer_per_thread:int, infer_result_queue:queue.Queue, input_blob, output_blob):


    # Sync with the main start barrier
    start_barrier.wait()
    
    # Start times for the fps counter
    start_time = time.time()
    end_time = start_time

    handle_list = [None]*simultaneous_infer_per_thread
    image_index = first_image_index
    image_result_start_index = 0

    inferences_per_req = int(num_total_inferences/simultaneous_infer_per_thread)
    # For each thread, 6 async inference requests will be created
    for outer_index in range(0, inferences_per_req):
        # Start the simultaneous async inferences
        for infer_id in range(0, simultaneous_infer_per_thread):
            new_request_id = dev_thread_request_id + infer_id
            handle_list[infer_id] = exec_net.start_async(request_id = new_request_id, inputs={input_blob: image_list[image_index]})
            image_index += 1
            if (image_index > last_image_index):
                image_index = first_image_index

        # Wait for the simultaneous async inferences to finish.
        for wait_index in range(0, simultaneous_infer_per_thread):
            infer_status = handle_list[wait_index].wait()
            result = handle_list[wait_index].outputs[output_blob]
            top_index = numpy.argsort(result, axis = 1)[0, -1:][::-1]
            top_index = top_index[0]
            prob = result[0][top_index]
            infer_result_queue.put((top_index, prob))

            handle_list[wait_index] = None



    # Save the time spent on inferences within this inference thread and associated reader thread
    end_time = time.time()
    total_inference_time = end_time - start_time
    result_list[result_index] = total_inference_time

    print("Thread " + str(result_index) + " end barrier reached")

    # Wait for all inference threads to finish
    end_barrier.wait()


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())
