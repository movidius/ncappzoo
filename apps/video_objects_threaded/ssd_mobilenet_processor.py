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


class ssd_mobilenet_processor:

    # Tiny Yolo assumes input images are these dimensions.
    SSDMN_NETWORK_IMAGE_WIDTH = 300
    SSDMN_NETWORK_IMAGE_HEIGHT = 300

    # initialize an instance of the class
    # tiny_yolo_graph_file is the path and filename to the tiny yolo graph
    #     file that was created by the ncsdk compiler
    # ncs_device is an open ncs device object
    # input_queue is a queue object from which images will be pulled and
    #     inferences will be processed on.
    # output_queue is a queue object on which the filtered inference results will
    #     be placed along with the input image as a tuple
    # The filtered results are a list of lists. each of the inner lists represent one found object and contain
    # the following 6 values:
    #    string that is network classification ie 'cat', or 'chair' etc
    #    float value for box X pixel location of upper left within source image
    #    float value for box Y pixel location of upper left within source image
    #    float value for box X pixel location of lower right within source image
    #    float value for box Y pixel location of lower right within source image
    #    float value that is the probability for the network classification 0.0 - 1.0 inclusive.
    # initial_box_prob_threshold is the initial box probability threshold for boxes
    #     returned from the inferences
    def __init__(self, network_graph_filename: str, ncs_device: mvnc.Device, input_queue:queue.Queue=None, output_queue: queue.Queue=None,
                 inital_box_prob_thresh: float=None, queue_wait_input:float=None, queue_wait_output:float=None):

        self._use_worker_thread = False
        self._queue_wait_input = queue_wait_input
        self._queue_wait_output = queue_wait_output

        # Load googlenet graph from disk and allocate graph via API
        try:
            with open(network_graph_filename, mode='rb') as graph_file:
                graph_in_memory = graph_file.read()
            self._graph = mvnc.Graph("SSD MobileNet Graph")
            self._fifo_in, self._fifo_out = self._graph.allocate_with_fifos(ncs_device, graph_in_memory)

        except:
            print('\n\n')
            print('Error - could not load tiny yolo graph file: ' + network_graph_filename)
            print('\n\n')
            raise

        self._box_probability_threshold = inital_box_prob_thresh

        self._input_queue = input_queue
        self._output_queue = output_queue
        self._end_flag = True
        if ( (not (self._queue_wait_input is None)) and
             (not (self._queue_wait_output is None)) and
             (not (self._input_queue is None)) and
             (not (self._output_queue is None)) ) :
            self._use_worker_thread = True

        self._worker_thread = threading.Thread(target=self._do_work, args=())

    # call once when done with the instance of the class
    def cleanup(self):

        self._drain_queues()
        self._fifo_in.destroy()
        self._fifo_out.destroy()
        self._graph.destroy()

    # start asynchronous processing of the images on the input queue via a worker thread
    # and place inference results on the output queue
    def start_processing(self):
        self._end_flag = False
        if (not self._use_worker_thread):
            return

        if (self._worker_thread == None):
            self._worker_thread = threading.Thread(target=self._do_work, args=())

        self._worker_thread.start()

    # stop asynchronous processing of the images on input queue
    # when returns the worker thread will be terminated
    def stop_processing(self):

        if (self._end_flag == True):
            # Already stopped
            return

        self._end_flag = True

        if (not self._use_worker_thread):
            return

        if (self._worker_thread == None):
            # no thread to wait for
            return

        self._worker_thread.join()
        self._worker_thread = None


    # Start an asynchronous inference.  When its complete it will go to the output FIFO queue which
    # can be read using the get_async_inference_result() method
    # If there is no room on the input queue this function will block indefinitely until there is room,
    # when there is room, it will queue the inference and return immediately
    #
    # input_image is the image on which to run the inference.
    #     it can be any size but is assumed to be opencv standard format of BGRBGRBGR...
    # Returns: nothing
    def start_aysnc_inference(self, input_image:numpy.ndarray):
        # save original width and height
        input_image_width = input_image.shape[1]
        input_image_height = input_image.shape[0]

        # resize image to network width and height
        # then convert to float32, normalize (divide by 255),
        # and finally convert to float16 to pass to LoadTensor as input
        # for an inference
        # this returns a new image so the input_image is unchanged
        inference_image = cv2.resize(input_image,
                                 (ssd_mobilenet_processor.SSDMN_NETWORK_IMAGE_WIDTH,
                                  ssd_mobilenet_processor.SSDMN_NETWORK_IMAGE_HEIGHT),
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

        # get the result from the queue
        output, input_image = self._fifo_out.read_elem()

        # save original width and height
        input_image_width = input_image.shape[1]
        input_image_height = input_image.shape[0]

        # filter out all the objects/boxes that don't meet thresholds
        return self._filter_objects(output, input_image_width, input_image_height), input_image


    # get the number of elemets in the input queue
    def is_input_queue_empty(self):
        if (self._use_worker_thread):
            ret_val = self._input_queue.empty()
        else:
            count = self._fifo_in.get_option(mvnc.FifoOption.RO_WRITE_FILL_LEVEL)
            retval = (count == 0)
        return count


    # clear everything from the input and output queues.
    # call this to clear both input and output queues after no longer putting work into the input queue
    def _drain_queues(self):

        #first clear the input queue
        if (self._use_worker_thread):
            self._input_queue.clear()
            self._output_queue.clear()
        else:
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

    # Do a single inference synchronously.
    # input_image is the image on which to run the inference.
    #     it can be any size
    # Returns: tuple of the filtered results along with the original input image
    # the filtered results is a list of lists. each of the inner lists represent one found object and contain
    # the following 6 values:
    #    string that is network classification ie 'cat', or 'chair' etc
    #    float value for box X pixel location of upper left within source image
    #    float value for box Y pixel location of upper left within source image
    #    float value for box X pixel location of lower right within source image
    #    float value for box Y pixel location of lower right within source image
    #    float value that is the probability for the network classification 0.0 - 1.0 inclusive.
    def do_sync_inference(self, input_image:numpy.ndarray):

        # save original width and height
        input_image_width = input_image.shape[1]
        input_image_height = input_image.shape[0]

        # resize image to network width and height
        # then convert to float32, normalize (divide by 255),
        # and finally convert to float16 to pass to LoadTensor as input
        # for an inference
        # this returns a new image so the input_image is unchanged
        inference_image = cv2.resize(input_image,
                                 (ssd_mobilenet_processor.SSDMN_NETWORK_IMAGE_WIDTH,
                                  ssd_mobilenet_processor.SSDMN_NETWORK_IMAGE_HEIGHT),
                                 cv2.INTER_LINEAR)

        # modify inference_image for network input
        inference_image = inference_image - 127.5
        inference_image = inference_image * 0.007843

        # Load tensor and get result.  This executes the inference on the NCS
        self._graph.queue_inference_with_fifo_elem(self._fifo_in, self._fifo_out, inference_image.astype(numpy.float32), input_image)
        output, userobj = self._fifo_out.read_elem()

        # until API is fixed to return mutable output.
        output.flags.writeable = True

        # filter out all the objects/boxes that don't meet thresholds
        return self._filter_objects(output, input_image_width, input_image_height)


    # the worker thread which handles the asynchronous processing of images on the input
    # queue, running inferences on the NCS and placing results on the output queue
    def _do_work(self):
        print('in ssd_mobilenet_processor worker thread')

        while (not self._end_flag):
            try:
                # get input image from input queue.  This does not copy the image
                input_image = self._input_queue.get(True, self._queue_wait_input)

                # get the inference and filter etc.
                filtered_objs = self.do_inference(input_image)

                # put the results along with the input image on the output queue
                self._output_queue.put((input_image, filtered_objs), True, self._queue_wait_output)

                # finished with this input queue work item
                self._input_queue.task_done()

            except queue.Empty:
                print('ssd_mobilenet_proc, input queue empty')
            except queue.Full:
                print('ssd_mobilenet_proc, output queue full')

        print('exiting ssd_mobilenet_processor worker thread')


    # get the box probability threshold.
    # will be between 0.0 and 1.0
    # higher number will result in less boxes returned
    # during inferences
    def get_box_probability_threshold(self):
        return self._box_probability_threshold

    # set the box probability threshold.
    # value is the new value, it must be between 0.0 and 1.0
    #     lower values will allow less certain boxes in the inferences
    #     which will result in more boxes per image.  Higher values will
    #     filter out less certain boxes and result in fewer boxes per
    #     inference.
    def set_box_probability_threshold(self, value):
        self._box_probability_threshold = value


    # Interpret the output from a single inference of the neural network
    # and filter out objects/boxes with low probabilities.
    # output is the array of floats returned from the API GetResult but converted
    # to float32 format.
    # input_image_width is the width of the input image
    # input_image_height is the height of the input image
    # Returns a list of lists. each of the inner lists represent one found object and contain
    # the following 6 values:
    #    string that is network classification ie 'cat', or 'chair' etc
    #    float value for box X pixel location of upper left within source image
    #    float value for box Y pixel location of upper left within source image
    #    float value for box X pixel location of lower right within source image
    #    float value for box Y pixel location of lower right within source image
    #    float value that is the probability for the network classification 0.0 - 1.0 inclusive.
    def _filter_objects(self, inference_result, input_image_width:int, input_image_height:int):

        # the raw number of floats returned from the inference
        num_inference_results = len(inference_result)

        # the 20 classes this network was trained on
        # labels AKA classes.  The class IDs returned
        # are the indices into this list
        network_classifications = ('background',
                  'aeroplane', 'bicycle', 'bird', 'boat',
                  'bottle', 'bus', 'car', 'cat', 'chair',
                  'cow', 'diningtable', 'dog', 'horse',
                  'motorbike', 'person', 'pottedplant',
                  'sheep', 'sofa', 'train', 'tvmonitor')

        # which types of objects do we want to include.
        #network_classifications_mask = [1, 1, 1, 1, 1, 1, 1,
        #                                1, 1, 1, 1, 1, 1, 1,
        #                                1, 1, 1, 1, 1, 1, 1]

        num_classifications = len(network_classifications) # should be 21


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

                classes_boxes_and_probs.append([network_classifications[int(inference_result[base_index + 1])], # label
                                                x1, y1, # upper left in source image
                                                x2, y2, # lower right in source image
                                                inference_result[base_index + 2] # confidence
                                                ])
        return classes_boxes_and_probs





