#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.

from openvino.inference_engine import IENetwork, IECore
import numpy as np
import cv2

class googlenet_processor:
    # set up some directories needed for the network
    EXAMPLES_BASE_DIR = '../../'
    ILSVRC_2012_dir = EXAMPLES_BASE_DIR + 'data/ilsvrc12/'
    MEAN_FILE_NAME = ILSVRC_2012_dir + 'ilsvrc_2012_mean.npy'
    LABELS_FILE_NAME = ILSVRC_2012_dir + 'synset_words.txt'
    
    # Constructor, takes an IR file string, ie core object, and a plugin/device name string
    def __init__(self, gn_ir: str, ie: IECore, device: str):
        self._ie = ie
        
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
        
        # Set up the network
        self._gn_net = IENetwork(model = gn_ir, weights = gn_ir[:-3] + 'bin')
        # Set up the input and output blobs
        self._gn_input_blob = next(iter(self._gn_net.inputs))
        self._gn_output_blob = next(iter(self._gn_net.outputs))
        self._gn_input_shape = self._gn_net.inputs[self._gn_input_blob].shape
        self._gn_output_shape = self._gn_net.outputs[self._gn_output_blob].shape
        # Load the network
        self._gn_exec_net = ie.load_network(network = self._gn_net, device_name = device)
        # Get the input shapes
        self._gn_n, self._gn_c, self._gn_h, self._gn_w = self._gn_input_shape
        
   
    def googlenet_inference(self, input_image):
        # Resize image to googlenet network width and height
        # Transpose the image to HWC, then perform a reshape
        input_image = cv2.resize(input_image, (self._gn_w, self._gn_h), cv2.INTER_LINEAR)
        input_image = np.transpose(input_image, (2,0,1))
        input_image = input_image.reshape((self._gn_n, self._gn_c, self._gn_h, self._gn_w))
        # convert to fp32
        input_image = input_image.astype(np.float32)
        
        # This executes the inference on the NCS
        self._gn_output = self._gn_exec_net.infer({self._gn_input_blob: input_image})
        # Sort the results and get the top index
        ret_index = np.argsort(self._gn_output[self._gn_output_blob], axis=1)[0, -1:][::-1]

        # Get the top label, and probability
        ret_label = self._gn_labels[ret_index[0]]
        ret_prob = self._gn_output[self._gn_output_blob][0, ret_index[0]]

        # check for inference results that are not birds.  The hardcoded numbers
        # in this condition are indices within the googlenet categories
        # read from the synset_words.txt file.  The inclusive ranges of indices
        # 127-147, 81-101, 7-24 are all known to be birds.  There may be other 
        # birds indices in which case they should be added here.
        if (not (((ret_index >= 127) and (ret_index <= 146)) or
                 ((ret_index >= 81) and (ret_index <=101)) or
                 ((ret_index >= 7) and (ret_index <=24)) )) :
            # its not classifying as a bird so set googlenet probability to 0
            ret_prob = 0.0
        return ret_index, ret_label, ret_prob
