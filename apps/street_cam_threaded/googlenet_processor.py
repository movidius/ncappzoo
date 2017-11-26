#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.
# NPS

# processes images via googlenet

from mvnc import mvncapi as mvnc
import numpy as np
import cv2
import queue
import threading


class googlenet_processor:
    # GoogLeNet assumes input images are these dimensions
    GN_NETWORK_IMAGE_WIDTH = 224
    GN_NETWORK_IMAGE_HEIGHT = 224

    EXAMPLES_BASE_DIR = '../../'
    ILSVRC_2012_dir = EXAMPLES_BASE_DIR + 'data/ilsvrc12/'

    MEAN_FILE_NAME = ILSVRC_2012_dir + 'ilsvrc_2012_mean.npy'

    LABELS_FILE_NAME = ILSVRC_2012_dir + 'synset_words.txt'

    # initialize the class instance
    # googlenet_graph_file is the path and filename of the googlenet graph file
    #     produced via the ncsdk compiler.
    # ncs_device is an open Device instance from the ncsdk
    # input_queue is a queue instance from which images will be pulled that are
    #     in turn processed (inferences are run on) via the NCS device
    #     each item on the queue should be an opencv image.  it will be resized
    #     as needed for the network
    # output_queue is a queue object on which the results of the inferences will be placed.
    #     For each inference a list of the following items will be placed on the output_queue:
    #         index of the most likely classification from the inference.
    #         label for the most likely classification from the inference.
    #         probability the most likely classification from the inference.
    def __init__(self, googlenet_graph_file, ncs_device, input_queue, output_queue,
                 queue_wait_input, queue_wait_output):

        self._queue_wait_input = queue_wait_input
        self._queue_wait_output = queue_wait_output

        # GoogLenet initialization

        # googlenet mean values will be read in from .npy file
        self._gn_mean = [0., 0., 0.]

        # labels to display along with boxes if googlenet classification is good
        # these will be read in from the synset_words.txt file for ilsvrc12
        self._gn_labels = [""]

        # loading the means from file
        try:
            self._gn_mean = np.load(googlenet_processor.MEAN_FILE_NAME).mean(1).mean(1)
        except:
            print('\n\n')
            print('Error - could not load means from ' + googlenet_processor.MEAN_FILE_NAME)
            print('\n\n')
            raise

        # loading the labels from file
        try:
            self._gn_labels = np.loadtxt(googlenet_processor.LABELS_FILE_NAME, str, delimiter='\t')
            for label_index in range(0, len(self._gn_labels)):
                temp = self._gn_labels[label_index].split(',')[0].split(' ', 1)[1]
                self._gn_labels[label_index] = temp
        except:
            print('\n\n')
            print('Error - could not read labels from: ' + googlenet_processor.LABELS_FILE_NAME)
            print('\n\n')
            raise

        # Load googlenet graph from disk and allocate graph via API
        try:
            with open(googlenet_graph_file, mode='rb') as gn_file:
                gn_graph_from_disk = gn_file.read()
            self._gn_graph = ncs_device.AllocateGraph(gn_graph_from_disk)

        except:
            print('\n\n')
            print('Error - could not load googlenet graph file: ' + googlenet_graph_file)
            print('\n\n')
            raise

        self._input_queue = input_queue
        self._output_queue = output_queue
        self._worker_thread = threading.Thread(target=self._do_work, args=())


    # call one time when the instance will no longer be used.
    def cleanup(self):
        self._gn_graph.DeallocateGraph()

    # start asynchronous processing on a worker thread that will pull images off the input queue and
    # placing results on the output queue
    def start_processing(self):
        self._end_flag = False
        if (self._worker_thread == None):
            self._worker_thread = threading.Thread(target=self._do_work, args=())
        self._worker_thread.start()

    # stop asynchronous processing of the worker thread.
    # when returns the worker thread will have terminated.
    def stop_processing(self):
        self._end_flag = True
        self._worker_thread.join()
        self._worker_thread = None

    # the worker thread function. called when start_processing is called and
    # returns when stop_processing is called.
    def _do_work(self):
        print('in googlenet_processor worker thread')
        while (not self._end_flag):
            try:
                input_image = self._input_queue.get(True, self._queue_wait_input)
                index, label, probability = self.googlenet_inference(input_image, "NPS")
                self._output_queue.put((index, label, probability), True, self._queue_wait_output)
                self._input_queue.task_done()
            except queue.Empty:
                print('googlenet processor: No more images in queue.')
            except queue.Full:
                print('googlenet processor: queue full')

        print('exiting googlenet_processor worker thread')


    # Executes an inference using the googlenet graph and image passed
    # gn_graph is the googlenet graph object to use for the inference
    #   its assumed that this has been created with allocate graph and the
    #   googlenet graph file on an open NCS device.
    # input_image is the image on which a googlenet inference should be
    #   executed.  It will be resized to match googlenet image size requirements
    #   and also converted to float32.
    # returns a list of the following three items
    #   index of the most likely classification from the inference.
    #   label for the most likely classification from the inference.
    #   probability the most likely classification from the inference.
    def googlenet_inference(self, input_image, user_obj):

        # Resize image to googlenet network width and height
        # then convert to float32, normalize (divide by 255),
        # and finally convert to convert to float16 to pass to LoadTensor as input for an inference
        input_image = cv2.resize(input_image, (googlenet_processor.GN_NETWORK_IMAGE_WIDTH,
                                               googlenet_processor.GN_NETWORK_IMAGE_HEIGHT),
                                 cv2.INTER_LINEAR)

        input_image = input_image.astype(np.float32)
        input_image[:, :, 0] = (input_image[:, :, 0] - self._gn_mean[0])
        input_image[:, :, 1] = (input_image[:, :, 1] - self._gn_mean[1])
        input_image[:, :, 2] = (input_image[:, :, 2] - self._gn_mean[2])

        # Load tensor and get result.  This executes the inference on the NCS
        self._gn_graph.LoadTensor(input_image.astype(np.float16), user_obj)
        output, userobj = self._gn_graph.GetResult()

        order = output.argsort()[::-1][:1]

        '''
        print('\n------- prediction --------')
        for i in range(0, 5):
            print('prediction ' + str(i) + ' (probability ' + str(output[order[i]]) + ') is ' + self._gn_labels[
                order[i]] + '  label index is: ' + str(order[i]))
        '''

        # index, label, probability
        return order[0], self._gn_labels[order[0]], output[order[0]]
