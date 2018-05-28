#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.
# NPS

# pulls images from video device and places them in a Queue
# if the queue is full will start to skip video frames.

#import numpy as np
import cv2
import queue
import threading
import time

class video_processor:

    # initializer for the class
    # Parameters:
    #   output_queue is an instance of queue.Queue in which the video frame
    #       images will be placed
    #   queue_put_wait_max is the max number of seconds to wait when putting
    #       images into the output queue.
    #   video_file is the video file to read
    #   queue_full_sleep_seconds is the number of seconds to sleep when the
    #       output queue is full.
    def __init__(self, video_file, request_video_width=640, request_video_height = 480,
                 obj_detect_proc=None, output_queue=None, queue_put_wait_max = 0.01,
                 queue_full_sleep_seconds = 0.1):
        self._queue_full_sleep_seconds = queue_full_sleep_seconds
        self._queue_put_wait_max = queue_put_wait_max
        self._video_file = video_file
        self._request_video_width = request_video_width
        self._request_video_height = request_video_height
        self._pause_mode = False

        # create the video device
        self._video_device = cv2.VideoCapture(self._video_file)

        if ((self._video_device == None) or (not self._video_device.isOpened())):
            print('\n\n')
            print('Error - could not open video device.')
            print('If you installed python opencv via pip or pip3 you')
            print('need to uninstall it and install from source with -D WITH_V4L=ON')
            print('Use the provided script: install-opencv-from_source.sh')
            print('\n\n')
            return

        # Request the dimensions
        self._video_device.set(cv2.CAP_PROP_FRAME_WIDTH, self._request_video_width)
        self._video_device.set(cv2.CAP_PROP_FRAME_HEIGHT, self._request_video_height)

        # save the actual dimensions
        self._actual_video_width = self._video_device.get(cv2.CAP_PROP_FRAME_WIDTH)
        self._actual_video_height = self._video_device.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print('actual video resolution: ' + str(self._actual_video_width) + ' x ' + str(self._actual_video_height))

        self._output_queue = output_queue
        self._obj_detect_proc = obj_detect_proc

        self._use_output_queue = False
        if (not(self._output_queue is None)):
            self._use_output_queue = True

        self._worker_thread = None #threading.Thread(target=self._do_work, args=())


    # the width of the images that will be put in the queue
    def get_actual_video_width(self):
        return self._actual_video_width

    # the height of the images that will be put in the queue
    def get_actual_video_height(self):
        return self._actual_video_height

    # start reading from the video and placing images in the output queue
    def start_processing(self):
        self._end_flag = False
        if (self._use_output_queue):
            if (self._worker_thread == None):
                self._worker_thread = threading.Thread(target=self._do_work_queue, args=())
        else:
            if (self._worker_thread == None):
                self._worker_thread = threading.Thread(target=self._do_work_obj_detect_proc, args=())

        self._worker_thread.start()

    # stop reading from video device and placing images in the output queue
    def stop_processing(self):
        if (self._end_flag == True):
            # Already stopped
            return

        self._end_flag = True


    def pause(self):
        self._pause_mode = True

    def unpause(self):
        self._pause_mode = False

    # Thread target.  When call start_processing and initialized with an output queue,
    # this function will be called in its own thread.  it will keep working until stop_processing is called.
    # or an error is encountered.
    def _do_work_queue(self):
        print('in video_processor worker thread')
        if (self._video_device == None):
            print('video_processor _video_device is None, returning.')
            return

        while (not self._end_flag):
            try:
                while (self._pause_mode):
                    time.sleep(0.1)

                ret_val, input_image = self._video_device.read()

                if (not ret_val):
                    print("No image from video device, exiting")
                    break
                self._output_queue.put(input_image, True, self._queue_put_wait_max)

            except queue.Full:
                # the video device is probably way faster than the processing
                # so if our output queue is full sleep a little while before
                # trying the next image from the video.
                time.sleep(self._queue_full_sleep_seconds)

        print('exiting video_processor worker thread for queue')


    # Thread target.  when call start_processing and initialized with an object detector processor,
    # this function will be called in its own thread.  it will keep working until stop_processing is called.
    # or an error is encountered.
    def _do_work_obj_detect_proc(self):
        print('in video_processor worker thread')
        if (self._video_device == None):
            print('video_processor _video_device is None, returning.')
            return

        while (not self._end_flag):
            try:
                while (self._pause_mode):
                    time.sleep(0.1)

                # Read from the video file
                ret_val, input_image = self._video_device.read()

                if (not ret_val):
                    print("No image from video device, exiting")
                    break

                self._obj_detect_proc.start_aysnc_inference(input_image)

            except Exception:
                # the video device is probably way faster than the processing
                # so if our output queue is full sleep a little while before
                # trying the next image from the video.
                print("Exception occurred writing to the neural network processor.")
                raise


        print('exiting video_processor worker thread for obj detection')

    # should be called once for each class instance when finished with it.
    def cleanup(self):
        # close video device

        # wait for worker thread to finish if it still exists
        if (not(self._worker_thread is None)):
            self._worker_thread.join()
            self._worker_thread = None

        self._video_device.release()
