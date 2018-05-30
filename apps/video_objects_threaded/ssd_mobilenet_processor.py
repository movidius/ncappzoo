#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.
# NPS

# Object detector using SSD Mobile Net

from mvnc import mvncapi as mvnc
import numpy as numpy
import cv2
import queue
import threading


class SsdMobileNetProcessor:

    # Neural network assumes input images are these dimensions.
    SSDMN_NETWORK_IMAGE_WIDTH = 300
    SSDMN_NETWORK_IMAGE_HEIGHT = 300


    def __init__(self, network_graph_filename: str, ncs_device: mvnc.Device,
                 inital_box_prob_thresh: float, classification_mask:list=None):
        """Initializes an instance of the class

        :param network_graph_filename: is the path and filename to the graph
               file that was created by the ncsdk compiler
        :param ncs_device: is an open ncs device object to use for inferences for this graph file
        :param inital_box_prob_thresh: the initial box probablity threshold. between 0.0 and 1.0
        :param classification_mask: a list of 0 or 1 values, one for each classification label in the
        _classification_mask list.  if the value is 0 then the corresponding classification won't be reported.
        :return : None
        """

        # Load graph from disk and allocate graph.
        try:
            with open(network_graph_filename, mode='rb') as graph_file:
                graph_in_memory = graph_file.read()
            self._graph = mvnc.Graph("SSD MobileNet Graph")
            self._fifo_in, self._fifo_out = self._graph.allocate_with_fifos(ncs_device, graph_in_memory)

        except:
            print('\n\n')
            print('Error - could not load neural network graph file: ' + network_graph_filename)
            print('\n\n')
            raise

        self._classification_labels=SsdMobileNetProcessor.get_classification_labels()

        self._box_probability_threshold = inital_box_prob_thresh
        self._classification_mask=classification_mask
        if (self._classification_mask is None):
            # if no mask passed then create one to accept all classifications
            self._classification_mask = [1, 1, 1, 1, 1, 1, 1,
                                         1, 1, 1, 1, 1, 1, 1,
                                         1, 1, 1, 1, 1, 1, 1]

        self._end_flag = True


    def cleanup(self):
        """Call once when done with the instance of the class

        :return: None
        """

        self._drain_queues()
        self._fifo_in.destroy()
        self._fifo_out.destroy()
        self._graph.destroy()


    @staticmethod
    def get_classification_labels():
        """get a list of the classifications that are supported by this neural network

        :return: the list of the classification strings
        """
        ret_labels = list(['background',
          'aeroplane', 'bicycle', 'bird', 'boat',
          'bottle', 'bus', 'car', 'cat', 'chair',
          'cow', 'dining table', 'dog', 'horse',
          'motorbike', 'person', 'potted plant',
          'sheep', 'sofa', 'train', 'tvmonitor'])
        return ret_labels


    def start_aysnc_inference(self, input_image:numpy.ndarray):
        """Start an asynchronous inference.  When its complete it will go to the output FIFO queue which
           can be read using the get_async_inference_result() method
           If there is no room on the input queue this function will block indefinitely until there is room,
           when there is room, it will queue the inference and return immediately

        :param input_image: he image on which to run the inference.
             it can be any size but is assumed to be opencv standard format of BGRBGRBGR...
        :return: None
        """

        # resize image to network width and height
        # then convert to float32, normalize (divide by 255),
        # and finally convert to float16 to pass to LoadTensor as input
        # for an inference
        # this returns a new image so the input_image is unchanged
        inference_image = cv2.resize(input_image,
                                 (SsdMobileNetProcessor.SSDMN_NETWORK_IMAGE_WIDTH,
                                  SsdMobileNetProcessor.SSDMN_NETWORK_IMAGE_HEIGHT),
                                 cv2.INTER_LINEAR)

        # modify inference_image for network input
        inference_image = inference_image - 127.5
        inference_image = inference_image * 0.007843

        # Load tensor and get result.  This executes the inference on the NCS
        self._graph.queue_inference_with_fifo_elem(self._fifo_in, self._fifo_out, inference_image.astype(numpy.float32), input_image)

        return


    # Reads the next available object from the output FIFO queue.
    # If there is nothing on the output FIFO, this fuction will block indefinitiley
    # until there is.
    # Returns tuple of the filtered results along with the original input image
    # the filtered results is a list of lists. each of the inner lists represent one found object and contain
    # the following 6 values:
    #    string that is network classification ie 'cat', or 'chair' etc
    #    float value for box X pixel location of upper left within source image
    #    float value for box Y pixel location of upper left within source image
    #    float value for box X pixel location of lower right within source image
    #    float value for box Y pixel location of lower right within source image
    #    float value that is the probability for the network classification 0.0 - 1.0 inclusive.
    def get_async_inference_result(self):
        """Reads the next available object from the output FIFO queue.  If there is nothing on the output FIFO,
        this fuction will block indefinitiley until there is.

        :return: tuple of the filtered results along with the original input image
        the filtered results is a list of lists. each of the inner lists represent one found object and contain
        the following 6 values:
           string that is network classification ie 'cat', or 'chair' etc
           float value for box X pixel location of upper left within source image
          float value for box Y pixel location of upper left within source image
          float value for box X pixel location of lower right within source image
          float value for box Y pixel location of lower right within source image
          float value that is the probability for the network classification 0.0 - 1.0 inclusive.
        """

        # get the result from the queue
        output, input_image = self._fifo_out.read_elem()

        # save original width and height
        input_image_width = input_image.shape[1]
        input_image_height = input_image.shape[0]

        # filter out all the objects/boxes that don't meet thresholds
        return self._filter_objects(output, input_image_width, input_image_height), input_image


    # get the number of elemets in the input queue
    def is_input_queue_empty(self):
        """ determines if the input queue for this instance is empty

        :return: True if input queue is empty or False if not.
        """
        count = self._fifo_in.get_option(mvnc.FifoOption.RO_WRITE_FILL_LEVEL)
        return (count == 0)


    def _drain_queues(self):
        """clears everything from the input and output queues. call this to clear both input and output
        queues after no longer putting work into the input queues  (calling start_async_inference)

        :return: None.
        """

        while (self._fifo_in.get_option(mvnc.FifoOption.RO_WRITE_FILL_LEVEL) != 0):
            # at least one item to process in the input queue, read one from output queue
            # and then loop back around
            print("input FIFO has at least one item")
            self._fifo_out.read_elem()

        while (self._fifo_out.get_option(mvnc.FifoOption.RO_READ_FILL_LEVEL) != 0):
            # output FIFO not empty so keep reading from it until it is
            print("output FIFO has at least one item")
            self._fifo_out.read_elem()

        print ("Done Draining queues")
        print ("Input FIFO fill level: " + str(self._fifo_in.get_option(mvnc.FifoOption.RO_WRITE_FILL_LEVEL)))
        print ("Output FIFO fill level: " + str(self._fifo_out.get_option(mvnc.FifoOption.RO_READ_FILL_LEVEL)))
        return


    def do_sync_inference(self, input_image:numpy.ndarray):
        """Do a single inference synchronously.
        Don't mix this with calls to get_async_inference_result, Use one or the other.  It is assumed
        that the input queue is empty when this is called which will be the case if this isn't mixed
        with calls to get_async_inference_result.

        :param input_image: the image on which to run the inference it can be any size.
        :return: filtered results which is a list of lists. Each of the inner lists represent one
        found object and contain the following 6 values:
            string that is network classification ie 'cat', or 'chair' etc
            float value for box X pixel location of upper left within source image
            float value for box Y pixel location of upper left within source image
            float value for box X pixel location of lower right within source image
            float value for box Y pixel location of lower right within source image
            float value that is the probability for the network classification 0.0 - 1.0 inclusive.
        """
        self.start_aysnc_inference(input_image)
        filtered_objects, original_image = self.get_async_inference_result()

        return filtered_objects


    def get_box_probability_threshold(self):
        """Determine the current box probabilty threshold for this instance.  It will be between 0.0 and 1.0.
        A higher number means less boxes will be returned.

        :return: the box probability threshold currently in place for this instance.
        """
        return self._box_probability_threshold


    def set_box_probability_threshold(self, value):
        """set the box probability threshold.

        :param value: the new box probability threshold value, it must be between 0.0 and 1.0.
        lower values will allow less certain boxes in the inferences
        which will result in more boxes per image.  Higher values will
        filter out less certain boxes and result in fewer boxes per
        inference.
        :return: None
        """
        self._box_probability_threshold = value


    def _filter_objects(self, inference_result:numpy.ndarray, input_image_width:int, input_image_height:int):
        """Interpret the output from a single inference of the neural network
        and filter out objects/boxes with probabilities under the box probability threshold

        :param inference_result: the array of floats returned from the NCAPI as float32.
        :param input_image_width: width of the original input image, used to determine size of boxes
        :param input_image_height: height of original input image used to determine size of boxes
        :return: list of lists. each of the inner lists represent one found object
        that match the filter criteria.  The each contain the following 6 values:
        string that is network classification ie 'cat', or 'chair' etc
            float value for box X pixel location of upper left within source image
            float value for box Y pixel location of upper left within source image
            float value for box X pixel location of lower right within source image
            float value for box Y pixel location of lower right within source image
            float value that is the probability for the network classification 0.0 - 1.0 inclusive.
        """

        # the inference result is in this format:
        #   a.	First value holds the number of valid detections = num_valid.
        #   b.	The next 6 values are unused.
        #   c.	The next (7 * num_valid) values contain the valid detections data
        #       Each group of 7 values will describe an object/box These 7 values in order.
        #       The values are:
        #         0: image_id (always 0)
        #         1: class_id (this is an index into labels)
        #         2: score (this is the probability for the class)
        #         3: box left location within image as number between 0.0 and 1.0
        #         4: box top location within image as number between 0.0 and 1.0
        #         5: box right location within image as number between 0.0 and 1.0
        #         6: box bottom location within image as number between 0.0 and 1.0

        # number of boxes returned
        num_valid_boxes = int(inference_result[0])

        classes_boxes_and_probs = []
        for box_index in range(num_valid_boxes):
                base_index = 7+ box_index * 7
                if (inference_result[base_index + 2] < self._box_probability_threshold):
                    # probability/confidence is too low for this box, omit it.
                    continue
                if (self._classification_mask[int(inference_result[base_index + 1])] != 1):
                    # masking off these types of objects
                    continue

                if (not numpy.isfinite(inference_result[base_index]) or
                        not numpy.isfinite(inference_result[base_index + 1]) or
                        not numpy.isfinite(inference_result[base_index + 2]) or
                        not numpy.isfinite(inference_result[base_index + 3]) or
                        not numpy.isfinite(inference_result[base_index + 4]) or
                        not numpy.isfinite(inference_result[base_index + 5]) or
                        not numpy.isfinite(inference_result[base_index + 6])):
                    # boxes with non finite (inf, nan, etc) numbers must be ignored
                    continue

                x1 = max(int(inference_result[base_index + 3] * input_image_width), 0)
                y1 = max(int(inference_result[base_index + 4] * input_image_height), 0)
                x2 = min(int(inference_result[base_index + 5] * input_image_width), input_image_width-1)
                y2 = min(int(inference_result[base_index + 6] * input_image_height), input_image_height-1)

                classes_boxes_and_probs.append([self._classification_labels[int(inference_result[base_index + 1])], # label
                                                x1, y1, # upper left in source image
                                                x2, y2, # lower right in source image
                                                inference_result[base_index + 2] # confidence
                                                ])
        return classes_boxes_and_probs





