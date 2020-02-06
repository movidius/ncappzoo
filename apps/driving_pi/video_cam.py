#! /usr/bin/env python3

# Copyright(c) 2020 Intel Corporation.
# License: MIT See LICENSE file in root directory.

from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import logging as log
import time
from itertools import count

res_width = 640
res_height = 480


class VideoCamera:
    """
    This class controling the camera, now it's supporting only Webcam and
    Camera-Pi camera. by default user is using 'usb' camera
    """

    def __init__(self, cam_type):
        self.cam_width = res_width
        self.cam_height = res_height
        self.camera_type = cam_type
        self.camera = None
        self.rawCapture = None

        if cam_type == "camerapi":
            log.info("Loading Camera Pi")
            self.camera = PiCamera()
            self.camera.resolution = (self.cam_width, self.cam_height)

        elif cam_type == "usb":
            camera_id = 0
            log.info("Loading USB Camera id {}".format(camera_id))
            self.camera = cv2.VideoCapture(camera_id)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)

        log.info("Camera size {}x{}".format(self.cam_width, self.cam_height))

    def frame(self):
        """
        Get frame from camera
        """
        if self.camera_type == "camerapi":
            self.rawCapture = PiRGBArray(self.camera, size=(self.cam_width, self.cam_height))
            framesArray = self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True)
            return next(framesArray).array

        elif self.camera_type == "usb":
            assert self.camera.isOpened(), "Couldn't open Camera"
            success, orig_frame = self.camera.read()
            assert success, "Can't snap image"
            return orig_frame

    def clean_video(self):
        """
        the cleaning the frame in each iteration of the loop.
        Use this function only when using camerapi type
        """
        if self.camera_type == "camerapi":
            self.rawCapture.truncate(0)
