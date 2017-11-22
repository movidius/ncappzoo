#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.
# NPS

# pulls images from camera device and places them in a Queue

#import numpy as np
import cv2
import queue
import threading

class camera_processor:

    # Requested and actual camera dimensions
    REQUEST_CAMERA_WIDTH = 640  # TY_NETWORK_IMAGE_WIDTH
    REQUEST_CAMERA_HEIGHT = 480  # TY_NETWORK_IMAGE_HEIGHT

    # Specifies which camera to use.  If only one it will likely be index 0
    CAMERA_INDEX = 0

    def __init__(self, output_queue):
        self._camera_device = cv2.VideoCapture(camera_processor.CAMERA_INDEX)

        if ((self._camera_device == None) or (not self._camera_device.isOpened())):
            print('\n\n')
            print('Error - could not open camera.  Make sure it is plugged in.')
            print('Also, if you installed python opencv via pip or pip3 you')
            print('need to uninstall it and install from source with -D WITH_V4L=ON')
            print('Use the provided script: install-opencv-from_source.sh')
            print('\n\n')
            return

        self._camera_device.set(cv2.CAP_PROP_FRAME_WIDTH, camera_processor.REQUEST_CAMERA_WIDTH)
        self._camera_device.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_processor.REQUEST_CAMERA_HEIGHT)

        self._actual_camera_width = self._camera_device.get(cv2.CAP_PROP_FRAME_WIDTH)
        self._actual_camera_height = self._camera_device.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print('actual camera resolution: ' + str(self._actual_camera_width) + ' x ' + str(self._actual_camera_height))

        self._output_queue = output_queue
        self._worker_thread = threading.Thread(target=self.do_work, args=())


    def get_actual_camera_width(self):
        return self._actual_camera_width

    def get_actual_camera_height(self):
        return self._actual_camera_height


    def start_processing(self):
        self._end_flag = False
        self._worker_thread.start()

    def stop_processing(self):
        self._end_flag = True

        '''
        NPS TODO: remove commented out code
        # remove one item off queue to allow the current put to finish
        # if the worker thread is blocked waiting to put
        try:
            input_image = self._output_queue.get(False)
            self._output_queue.task_done()
        except:
            print('handling exception')
            pass
        '''
        self._worker_thread.join()

    def do_work(self):
        print('in camera_processor worker thread')
        if (self._camera_device == None):
            print('camera_processor camera_device is None, returning.')
            return
        while (not self._end_flag):
            try:
                ret_val, input_image = self._camera_device.read()
                if (not ret_val):
                    print("No image from camera, exiting")
                    break
                #print('camera_processor got image')
                self._output_queue.put(input_image, True, 4)
            except queue.Full:
                print('camera_proc output queue full.')

        print('exiting camera_processor worker thread')

    def cleanup(self):
        # close camera
        self._camera_device.release()
