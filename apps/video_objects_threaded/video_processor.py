#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.
# NPS

# pulls images from video device and places them in a Queue or starts an inference for them on a network processor

import cv2
import queue
import threading
import time
from ssd_mobilenet_processor import SsdMobileNetProcessor
from queue import Queue

class VideoProcessor:
    """Class that pulls frames from a video file and either starts an inference with them or
    puts them on a queue depending on how the instance is constructed.
    """

    def __init__(self, video_file:str, request_video_width:int=640, request_video_height:int = 480,
                 network_processor:SsdMobileNetProcessor=None, output_queue:Queue=None, queue_put_wait_max:float = 0.01,
                 queue_full_sleep_seconds:float = 0.1):
        """Initializer for the class.

        :param video_file: file name of the file from which to read video frames
        :param request_video_width: the width in pixels to request from the video device, may be ignored.
        :param request_video_height: the height in pixels to request from the video device, may be ignored.
        :param network_processor: neural network processor on which we will start inferences for each frame.
        If a value is passed
        for this parameter then the output_queue, queue_put_wait_max, and queue_full_sleep_seconds will be ignored
        and should be None
        :param output_queue: A queue on which the video frames will be placed if the network_processor is None
        :param queue_put_wait_max: The max number of seconds to wait when putting on output queue
        :param queue_full_sleep_seconds: The number of seconds to sleep when the output queue is full.
        """
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
        self._network_processor = network_processor

        self._use_output_queue = False
        if (not(self._output_queue is None)):
            self._use_output_queue = True

        self._worker_thread = None #threading.Thread(target=self._do_work, args=())


    def get_actual_video_width(self):
        """ get the width of the images that will be placed on queue or sent to neural network processor.
        :return: the width of each frame retrieved from the video device
        """
        return self._actual_video_width

    # the
    def get_actual_video_height(self):
        """get the height of the images that will be put in the queue or sent to the neural network processor

        :return: The height of each frame retrieved from the video device
        """
        return self._actual_video_height


    def start_processing(self):
        """Starts the asynchronous thread reading from the video file and placing images in the output queue or sending to the
        neural network processor

        :return: None
        """
        self._end_flag = False
        if (self._use_output_queue):
            if (self._worker_thread == None):
                self._worker_thread = threading.Thread(target=self._do_work_queue, args=())
        else:
            if (self._worker_thread == None):
                self._worker_thread = threading.Thread(target=self._do_work_network_processor, args=())

        self._worker_thread.start()


    def stop_processing(self):
        """stops the asynchronous thread from reading any new frames from the video device

        :return:
        """
        if (self._end_flag == True):
            # Already stopped
            return

        self._end_flag = True


    def pause(self):
        """pauses the aysnchronous processing so that it will not read any new frames until unpause is called.
        :return: None
        """
        self._pause_mode = True



    def unpause(self):
        """ Unpauses the asynchronous processing that was previously paused by calling pause

        :return: None
        """
        self._pause_mode = False

    # Thread target.  When call start_processing and initialized with an output queue,
    # this function will be called in its own thread.  it will keep working until stop_processing is called.
    # or an error is encountered.
    def _do_work_queue(self):
        """Thread target.  When call start_processing and initialized with an output queue,
           this function will be called in its own thread.  it will keep working until stop_processing is called.
           or an error is encountered.  If the neural network processor was passed to the initializer rather than
           a queue then this function will not be called.

        :return: None
        """
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


    def _do_work_network_processor(self):
        """Thread target.  when call start_processing and initialized with an neural network processor,
           this function will be called in its own thread.  it will keep working until stop_processing is called.
           or an error is encountered.  If the initializer was called with a queue rather than a neural network
           processor then this will not be called.

        :return: None
        """
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

                self._network_processor.start_aysnc_inference(input_image)

            except Exception:
                # the video device is probably way faster than the processing
                # so if our output queue is full sleep a little while before
                # trying the next image from the video.
                print("Exception occurred writing to the neural network processor.")
                raise


        print('exiting video_processor worker thread for network processor')


    def cleanup(self):
        """Should be called once for each class instance when the class consumer is finished with it.

        :return: None
        """
        # wait for worker thread to finish if it still exists
        if (not(self._worker_thread is None)):
            self._worker_thread.join()
            self._worker_thread = None

        self._video_device.release()
