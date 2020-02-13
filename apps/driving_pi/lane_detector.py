#! /usr/bin/env python3

# Copyright(c) 2020 Intel Corporation.
# License: MIT See LICENSE file in root directory.

import cv2
import numpy as np
from collections import deque


def convert_hsv(image):
    """
    Converting frame to HSV - see more detsils on OpenCV
    :param image: frame image
    :return: coverted image to HSV   
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def convert_gray_scale(image):
    """
    Converting frame to GRAY - see more detsils on OpenCV
    :param image: frame image
    :return: coverted image to GRAY
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def add_gaussian_blur(image, kernel_size=5):
    """
    Make image more blur which will make more the performance much better
    :param image: frame image
    :param kernel_size: |int| kernel_size must be positive and odd
    :return: blur image
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def detect_edges(image, low_threshold=50, high_threshold=150):
    """
    Using Canny function for getting the edges of the lanes on the road
    :param image: frame image
    :param low_threshold: |int| low threshold
    :param high_threshold:  |int| high threshold
    :return: detected lane edges
    """
    return cv2.Canny(image, low_threshold, high_threshold)


def select_yellow(image):
    """
    Get the yellow color only for the frame!!
    IMPORTANT: if you're using different road please change the color (of the middle line)
    :param image: frame image
    :return: a frame with yellow color only (the middle line)
    """
    converted = convert_hsv(image)
    # yellow color mask
    lower = np.array([18, 94, 140])
    upper = np.array([48, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    return yellow_mask


def hough_lines(image):
    """
    Creates hough line
    Note that: `image` should be the output of a Canny transform.
    :param image: |Image| camera frame
    :return: hough lines (not the image with lines)
    """
    return cv2.HoughLinesP(image, 2, np.pi / 180, 50, maxLineGap=50)


def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        # in case, the input image has a channel dimension
        cv2.fillPoly(mask, vertices, (255,) * mask.shape[2])

    return cv2.bitwise_and(image, mask)


def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  
    Other area is set to 0 (black).
    :param image: |Image| camera frame
    :return: image with street region only
    """
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.3, rows * 0.3]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.7, rows * 0.3]
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    return filter_region(image, vertices)


class LaneDetector:
    """
    Creating class to control the lanes on the street.
    on this class, the camera looking for the Yellow color on the middle
    of the street. in this way, the car start moving forward, change this color
    base on the street you run the car on it.
    """

    def __init__(self, image):
        self.image = image

    def get_lane(self):
        """
        Get the middle line of the street.
        """
        smooth_gray = add_gaussian_blur(self.image)
        yellow_line = select_yellow(smooth_gray)
        detect_edge = detect_edges(yellow_line, low_threshold=75, high_threshold=150)
        hough_line = hough_lines(detect_edge)

        return hough_line

    @property
    def image(self):
        """
        Get image frame from the camera
        :return: frame
        """
        return self.__image

    @image.setter
    def image(self, frame):
        """
        Set new frame and replace it as a new iamge
        :param frame: |Image| camera frame
        """
        self.__image = frame
