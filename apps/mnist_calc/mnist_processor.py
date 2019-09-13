#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.
# NPS, Heather McCabe
# Digit classifier using MNIST

import cv2
import numpy
from openvino.inference_engine import IENetwork, IEPlugin

class MnistProcessor:
    # The network assumes input images are these dimensions
    NETWORK_IMAGE_WIDTH = 28
    NETWORK_IMAGE_HEIGHT = 28

    def __init__(self, network_graph_filename, plugin=None,
                 prob_thresh=0.0, classification_mask=None):
        """Initialize an instance of the class.

        :param network_graph_filename: the file path and name of the graph file created by the OpenVINO compiler
        :param nc_device: an open neural compute device object to use for inferences for this graph file
        :param prob_thresh: the probability threshold (between 0.0 and 1.0)... results below this threshold will be
        excluded
        :param classification_mask: a list of 0/1 values, one for each classification label in the
        _classification_mask list... if the value is 0 then the corresponding classification won't be reported.
        :return : None

        """
        # Load graph from disk and allocate graph to device
        try:
            
            self._net = IENetwork(model=network_graph_filename + ".xml", weights=network_graph_filename + ".bin")
            self._input_blob = next(iter(self._net.inputs))
            self._output_blob = next(iter(self._net.outputs))
            self._exec_net = plugin.load(network=self._net)
            #print("network shape: ",self._net.inputs[self._input_blob].shape)
            self._n, self._input_total_size = self._net.inputs[self._input_blob].shape
            del self._net


        except IOError as e:
            print('Error - could not load neural network graph file: ' + network_graph_filename)
            raise e

        # If no mask was passed then create one to accept all classifications
        self._classification_mask = classification_mask if classification_mask else [1] * 10
        self._probability_threshold = prob_thresh
        self._end_flag = True


    def _process_results(self, inference_result):
        """Interpret the output from a single inference of the neural network and filter out results with
        probabilities under the probability threshold.

        :param inference_result: the array of floats returned from the NCAPI as float32
        :return results: a list of 2-element sublists containing the detected digit as a string and the associated
        probability as a float

        """
        results = []
        # Get a list of inference_result indexes sorted from highest to lowest probability
        sorted_indexes = (-inference_result[0]).argsort()
        results.append([sorted_indexes[0], inference_result[0][sorted_indexes[0]]])
        
        return results

    def cleanup(self):
        """Clean up mvncapi objects. Call once when done with the instance of the class.

        :return: None
        """
        del self._exec_net

    @staticmethod
    def get_classification_labels():
        """Get a list of the classifications that are supported by this neural network.

        :return: the list of the classification strings
        """
        return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


    def start_async_inference(self, input_image):
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
        inference_image = inference_image.reshape((self._n, self._input_total_size))
        
        # Start the async inference
        self._req_handle = self._exec_net.start_async(request_id=0, inputs={self._input_blob: inference_image})
        
    def get_async_inference_result(self):
        """Read the next available result from the output FIFO queue. If there is nothing on the output FIFO,
        this function will block indefinitely until there is something to read.

        :return: tuple of the filtered results and the original input image

        """
        # Get the result from the queue
        self._status = self._req_handle.wait()
        output = self._req_handle.outputs[self._output_blob]
        
        # Get a ranked list of results that meet the probability thresholds
        return self._process_results(output)

    def do_async_inference(self, input_image: numpy.ndarray):
        """Do a single inference synchronously.

        Don't mix this with calls to get_async_inference_result. Use one or the other. It is assumed
        that the input queue is empty when this is called which will be the case if this isn't mixed
        with calls to get_async_inference_result.

        :param input_image: the image on which to run the inference - it can be any size.
        :return: filtered results which is a list of lists. The inner lists contain the digit and its probability and
        are sorted from most probable to least probable.
        """
        self.start_async_inference(input_image)
        results = self.get_async_inference_result()

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
