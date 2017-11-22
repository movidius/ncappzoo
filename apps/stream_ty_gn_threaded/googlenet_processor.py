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

    def __init__(self, googlenet_graph_file, ncs_device, input_queue, output_queue):

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
        self._worker_thread = threading.Thread(target=self.do_work, args=())


    def cleanup(self):
        self._gn_graph.DeallocateGraph()


    def start_processing(self):
        self._end_flag = False
        self._worker_thread.start()


    def stop_processing(self):
        self._end_flag = True
        '''
        NPS TODO: remove commented code
        # remove empty the input queue
        try:
            while (self._input_queue.not_empty()):
                self._input_queue.get(False)
                self._input_queue.task_done()
        except:
            print('gn_proc input, handling exception')
            pass

        try:
            while (self._output_queue.not_empty()):
                self._output_queue.get(False)
                self._output_queue.task_done()
        except:
            print('gn_proc output, handling exception')
            pass
        '''
        self._worker_thread.join()


    def do_work(self):
        print('in googlenet_processor worker thread')
        while (not self._end_flag):
            try:
                input_image = self._input_queue.get(True, 4)
                index, label, probability = self.googlenet_inference(input_image, "NPS")
                self._output_queue.put((index, label, probability), True, 4)
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
