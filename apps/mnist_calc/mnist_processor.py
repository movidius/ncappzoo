#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.
# NPS, Heather McCabe
# Digit classifier using MNIST

import cv2
from mvnc import mvncapi as mvnc
import numpy


class MnistProcessor:
    # The network assumes input images are these dimensions
    NETWORK_IMAGE_WIDTH = 28
    NETWORK_IMAGE_HEIGHT = 28

    def __init__(self, network_graph_filename, nc_device=None,
                 prob_thresh=0.0, classification_mask=None):
        """Initialize an instance of the class.

        :param network_graph_filename: the file path and name of the graph file created by the NCSDK compiler
        :param nc_device: an open neural compute device object to use for inferences for this graph file
        :param prob_thresh: the probability threshold (between 0.0 and 1.0)... results below this threshold will be
        excluded
        :param classification_mask: a list of 0/1 values, one for each classification label in the
        _classification_mask list... if the value is 0 then the corresponding classification won't be reported.
        :return : None

        """
        # Load graph from disk and allocate graph to device
        try:
            with open(network_graph_filename, mode='rb') as graph_file:
                graph_in_memory = graph_file.read()
            self._graph = mvnc.Graph("MNIST Graph")
            self._fifo_in, self._fifo_out = self._graph.allocate_with_fifos(nc_device, graph_in_memory)

        except IOError as e:
            print('Error - could not load neural network graph file: ' + network_graph_filename)
            raise e

        # If no mask was passed then create one to accept all classifications
        self._classification_mask = classification_mask if classification_mask else [1] * 10
        self._probability_threshold = prob_thresh
        self._end_flag = True

    def _drain_queues(self):
        """Clear everything from the input and output queues.

        :return: None.
        """

        # Clear out the input queue
        while self._fifo_in.get_option(mvnc.FifoOption.RO_WRITE_FILL_LEVEL) != 0:
            # if at least one item to process in the input queue, read one from output queue to make room
            #print("input FIFO has at least one item")
            self._fifo_out.read_elem()

        while self._fifo_out.get_option(mvnc.FifoOption.RO_READ_FILL_LEVEL) != 0:
            # output FIFO not empty so keep reading from it until it is
            #print("output FIFO has at least one item")
            self._fifo_out.read_elem()

        #print("Done draining queues")
        #print("Input FIFO fill level: " + str(self._fifo_in.get_option(mvnc.FifoOption.RO_WRITE_FILL_LEVEL)))
        #print("Output FIFO fill level: " + str(self._fifo_out.get_option(mvnc.FifoOption.RO_READ_FILL_LEVEL)))
        return

    def _process_results(self, inference_result):
        """Interpret the output from a single inference of the neural network and filter out results with
        probabilities under the probability threshold.

        :param inference_result: the array of floats returned from the NCAPI as float32
        :return results: a list of 2-element sublists containing the detected digit as a string and the associated
        probability as a float

        """
        results = []

        # Get a list of inference_result indexes sorted from highest to lowest probability
        sorted_indexes = (-inference_result).argsort()

        # Get a list of sub-lists containing the detected digit as an int and the probability
        for i in sorted_indexes:
            if inference_result[i] >= self._probability_threshold:
                results.append([i, inference_result[i]])
            else:
                # If this index had a value under the probability threshold, the rest of the indexes will too
                break

        return results

    def cleanup(self):
        """Clean up mvncapi objects. Call once when done with the instance of the class.

        :return: None
        """

        self._drain_queues()
        self._fifo_in.destroy()
        self._fifo_out.destroy()
        self._graph.destroy()

    @staticmethod
    def get_classification_labels():
        """Get a list of the classifications that are supported by this neural network.

        :return: the list of the classification strings
        """
        return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    def is_input_queue_empty(self):
        """Determine if the input queue for this instance is empty.

        :return: True if input queue is empty or False if not.
        """
        count = self._fifo_in.get_option(mvnc.FifoOption.RO_WRITE_FILL_LEVEL)
        return count == 0

    def start_aysnc_inference(self, input_image):
        """Start an asynchronous inference.

        When the inference complete the result will go to the output FIFO queue, which can then be read using the
        get_async_inference_result() method.

        If there is no room on the input queue this function will block indefinitely until there is room;
        when there is room, it will queue the inference and return immediately.

        :param input_image: the image on which to run the inference.
             This can be any size but is assumed to be in the OpenCV standard format of BGRBGRBGR...
        :return: None
        """
        # Convert the image to binary black and white
        inference_image = cv2.bitwise_not(input_image)
        inference_image = cv2.cvtColor(inference_image, cv2.COLOR_BGR2GRAY)

        # Make the image square by creating a new square_img and copying inference_img into its center
        h, w = inference_image.shape
        h_diff = w - h if w > h else 0
        w_diff = h - w if h > w else 0
        square_img = numpy.zeros((w + w_diff, h + h_diff), numpy.uint8)
        square_img[int(h_diff / 2): int(h_diff / 2) + h, int(w_diff / 2): int(w_diff/2) + w] = inference_image
        inference_image = square_img

        # Resize the image
        padding = 2
        inference_image = cv2.resize(inference_image,
                                     (MnistProcessor.NETWORK_IMAGE_WIDTH - padding * 2,
                                      MnistProcessor.NETWORK_IMAGE_HEIGHT - padding * 2),
                                     cv2.INTER_LINEAR)

        # Pad the edges slightly to make sure the number isn't bumping against the edges
        inference_image = numpy.pad(inference_image, (padding, padding), 'constant', constant_values=0)

        # Modify inference_image for network input
        inference_image[:] = ((inference_image[:]) * (1.0 / 255.0))

        # Load tensor and get result.  This executes the inference on the NCS
        self._graph.queue_inference_with_fifo_elem(self._fifo_in, self._fifo_out,
                                                   inference_image.astype(numpy.float32), input_image)

    def get_async_inference_result(self):
        """Read the next available result from the output FIFO queue. If there is nothing on the output FIFO,
        this function will block indefinitely until there is something to read.

        :return: tuple of the filtered results and the original input image

        """
        # Get the result from the queue
        output, input_image = self._fifo_out.read_elem()

        # Get a ranked list of results that meet the probability thresholds
        return self._process_results(output), input_image

    def do_sync_inference(self, input_image: numpy.ndarray):
        """Do a single inference synchronously.

        Don't mix this with calls to get_async_inference_result. Use one or the other. It is assumed
        that the input queue is empty when this is called which will be the case if this isn't mixed
        with calls to get_async_inference_result.

        :param input_image: the image on which to run the inference - it can be any size.
        :return: filtered results which is a list of lists. The inner lists contain the digit and its probability and
        are sorted from most probable to least probable.
        """
        self.start_aysnc_inference(input_image)
        results, original_image = self.get_async_inference_result()

        return results

    @property
    def probability_threshold(self):
        return self._probability_threshold

    @probability_threshold.setter
    def probability_threshold(self, value):
        if 0.0 <= value <= 1.0:
            self._probability_threshold = value
        else:
            raise AttributeError('probability_threshold must be in range 0.0-1.0')
