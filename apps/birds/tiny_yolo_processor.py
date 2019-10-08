#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.

from openvino.inference_engine import IENetwork, IECore
import numpy as np
import cv2

class tiny_yolo_processor:
    # Set up parameters that will be needed for tiny yolo
    EXAMPLES_BASE_DIR = '../../'
    DETECTION_THRESHOLD = 0.10
    MAX_IOU = 0.25
    LABELS_FILE_NAME = 'labels.txt'
    
    # Initialize the class instance
    # ty_ir is the tiny yolo intermediate representation file
    # ie is the inference core object
    # device is the name of the plugin/device to use for inference
    # also sets up the network input/output
    def __init__(self, ty_ir: str, ie: IECore, device: str):
        self._ie = ie
        
        try:
            self._ty_labels = np.loadtxt(tiny_yolo_processor.LABELS_FILE_NAME, str, delimiter='\t')
            with open(tiny_yolo_processor.LABELS_FILE_NAME) as labels_file:
	            self._ty_labels = labels_file.read().splitlines()
        except:
            print('\n\n')
            print('Error - could not read labels from: ' + tiny_yolo_processor.LABELS_FILE_NAME)
            print('\n\n')
            raise
        
        # Create the network object
        self._ty_net = IENetwork(model = ty_ir, weights = ty_ir[:-3] + 'bin')
        # Set up the input and output blobs
        self._ty_input_blob = next(iter(self._ty_net.inputs))
        self._ty_output_blob = next(iter(self._ty_net.outputs))
        self._ty_input_shape = self._ty_net.inputs[self._ty_input_blob].shape
        self._ty_output_shape = self._ty_net.outputs[self._ty_output_blob].shape
        # Load the network
        self._ty_exec_net = ie.load_network(network = self._ty_net, device_name = device)
        # Get the input shape
        self._ty_n, self._ty_c, self.ty_h, self.ty_w = self._ty_input_shape

    
    # Performs the image preprocessing and makes an inference
    # Returns a list of detected object names, four bounding box points, and the object score
    def tiny_yolo_inference(self, input_image):
        # Resize the image, convert to fp32, then transpose to CHW
        input_image = cv2.resize(input_image, (self.ty_w, self.ty_h), cv2.INTER_LINEAR)
        input_image = input_image.astype(np.float32)
        input_image = np.transpose(input_image, (2,0,1))

        # Performs the inference
        self._ty_output = self._ty_exec_net.infer({self._ty_input_blob: input_image})
        # Flattens the output results for the output node
        self._ty_output = self._ty_output[self._ty_output_blob][0].flatten()
        
        # Filter out results based on iou, score
        self._filtered_objs = self._filter_objects(input_image.shape[1], input_image.shape[2]) 
        return self._filtered_objs
        

    # Interpret the output from a single inference of TinyYolo (GetResult)
    # and filter out objects/boxes with low probabilities.
    # output is the array of floats returned from the API GetResult but converted
    # to float32 format.
    # input_image_width is the width of the input image
    # input_image_height is the height of the input image
    # Returns a list of lists. each of the inner lists represent one found object and contain
    # the following 6 values:
    #    string that is network classification ie 'cat', or 'chair' etc
    #    float value for box center X pixel location within source image
    #    float value for box center Y pixel location within source image
    #    float value for box width in pixels within source image
    #    float value for box height in pixels within source image
    #    float value that is the probability for the network classification.
    def _filter_objects(self, input_image_width, input_image_height):

        # tiny yolo v1 was trained using a 7x7 grid and 2 anchor boxes per grid box with 
        # 20 detection classes
        # the 20 classes this network was trained on

        num_classes = len(self._ty_labels) # should be 20

        grid_size = 7 # the image is a 7x7 grid.  Each box in the grid is 64x64 pixels
        anchor_boxes_per_grid_cell = 2 # the number of anchor boxes returned for each grid cell
        num_coordinates = 4 # number of coordinates

        # the raw number of floats returned from the inference
        num_inference_results = len(self._ty_output)

        # only keep boxes with probabilities greater than this
        self._probability_threshold = tiny_yolo_processor.DETECTION_THRESHOLD
       
        # -------------------- Inference result preprocessing --------------------
        # Split the Inference result into 3 arrays: class_probabilities, box_confidence_scores, box_coordinates
        # then Reshape them into the appropriate shapes.
        
        # -- Splitting up Inference result
        
        # Class probabilities: 
        # 7x7 = 49 grid cells. 
        # 49 grid cells x 20 classes per grid cell = 980 total class probabilities
        class_probabilities = self._ty_output[0:980]
        
        # Box confidence scores: 7x7 = 49 grid cells. "how likely the box contains an object" 
        # 49 grid cells x 2 boxes per grid cell = 98 box scales
        box_confidence_scores = self._ty_output[980:1078]
        
        # Box coordinates for all boxes 
        # 98 boxes * 4 box coordinates each = 392
        box_coordinates = self._ty_output[1078:]
        
        # -- Reshaping 
        
        # These values are the class probabilities for each grid
        # Reshape the probabilities to 7x7x20 (980 total values)
        class_probabilities = np.reshape(class_probabilities, (grid_size, grid_size, num_classes))

        # These values are how likely each box contains an object
        # Reshape the box confidence scores to 7x7x2 (98 total values)
        box_confidence_scores = np.reshape(box_confidence_scores, (grid_size, grid_size, anchor_boxes_per_grid_cell))

        # These values are the box coordinates for each box
        # Reshape the boxes coordinates to 7x7x2x4 (392 total values)
        box_coordinates = np.reshape(box_coordinates, (grid_size, grid_size, anchor_boxes_per_grid_cell, num_coordinates))
        
        # -------------------- Scale the box coordinates to the input image size --------------------
        self._boxes_to_pixel_units(box_coordinates, input_image_width, input_image_height, grid_size)


        # -------------------- Calculate class confidence scores --------------------
        # Find the class confidence scores for each grid. 
        # This is done by multiplying the class probabilities by the box confidence scores 
        # Shape of class confidence scores: 7x7x2x20 (1960 values)
        class_confidence_scores = np.zeros((grid_size, grid_size, anchor_boxes_per_grid_cell, num_classes))
        for box_index in range(anchor_boxes_per_grid_cell): # loop over boxes
            for class_index in range(num_classes): # loop over classifications
                class_confidence_scores[:,:,box_index,class_index] = np.multiply(class_probabilities[:,:,class_index], box_confidence_scores[:,:,box_index])

        
        # -------------------- Filter object scores/coordinates/indexes >= threshold --------------------
        # Find all scores that are larger than or equal to the threshold using a mask.
        # Array of 1960 bools: True if >= threshold. otherwise False. 
        score_threshold_mask = np.array(class_confidence_scores>=self._probability_threshold, dtype='bool')
        # Using the array of bools, filter all scores >= threshold
        filtered_scores = class_confidence_scores[score_threshold_mask]
        
        # Get tuple of arrays of indexes from the bool array that have a >= score than the threshold
        # These tuple of array indexes will help to filter out our box coordinates and class indexes 
        # tuple 0 and 1 are the coordinates of the 7x7 grid (values = 0-6)
        # tuple 2 is the anchor box index (values = 0-1)
        # tuple 3 is the class indexes (labels) (values = 0-19)
        box_threshold_mask = np.nonzero(score_threshold_mask)
        
        # Use those indexes to find the coordinates for box confidence scores >= than the threshold
        filtered_box_coordinates = box_coordinates[box_threshold_mask[0], box_threshold_mask[1], box_threshold_mask[2]]
        
        # Use those indexes to find the class indexes that have a score >= threshold 
        filtered_class_indexes = np.argmax(class_confidence_scores, axis=3)[box_threshold_mask[0], box_threshold_mask[1], box_threshold_mask[2]]
        
        # -------------------- Sort the filtered scores/coordinates/indexes --------------------
        # Sort the indexes from highest score to lowest 
        # and then use those indexes to sort box coordinates, scores, class indexes
        sort_by_highest_score = np.array(np.argsort(filtered_scores))[::-1]
        # Sort the box coordinates, scores, and class indexes to match 
        filtered_box_coordinates = filtered_box_coordinates[sort_by_highest_score]
        filtered_scores = filtered_scores[sort_by_highest_score]    
        filtered_class_indexes = filtered_class_indexes[sort_by_highest_score]

        # -------------------- Filter out duplicates --------------------
        # Get mask for boxes that seem to be the same object by calculating iou (intersection over union)
        # these will filter out duplicate objects
        duplicate_box_mask = self._get_duplicate_box_mask(filtered_box_coordinates)
        # Update the boxes, probabilities and classifications removing duplicates.
        filtered_box_coordinates = filtered_box_coordinates[duplicate_box_mask]
        filtered_scores = filtered_scores[duplicate_box_mask]
        filtered_class_indexes = filtered_class_indexes[duplicate_box_mask]
        
        # -------------------- Gather the results --------------------
        # Set up list and return class labels, coordinates and scores
        filtered_results = []
        for object_index in range( len( filtered_box_coordinates ) ):
            filtered_results.append([
            self._ty_labels [ filtered_class_indexes [ object_index ] ], # label of the object
            filtered_box_coordinates [ object_index ] [ 0 ],    # xmin (before image scaling)
            filtered_box_coordinates [ object_index ] [ 1 ],    # ymin (before image scaling)
            filtered_box_coordinates [ object_index ] [ 2 ],    # width (before image scaling)
            filtered_box_coordinates [ object_index ] [ 3 ],    # height (before image scaling)
            filtered_scores [ object_index ]                    # object score
            ])

        return filtered_results


    # creates a mask to remove duplicate objects (boxes) and their related probabilities and classifications
    # that should be considered the same object.  This is determined by how similar the boxes are
    # based on the intersection-over-union metric.
    # box_list is as list of boxes (4 floats for centerX, centerY and Length and Width)
    def _get_duplicate_box_mask(self, box_list):
        # The intersection-over-union threshold to use when determining duplicates.
        # objects/boxes found that are over this threshold will be
        # considered the same object
        max_iou = tiny_yolo_processor.MAX_IOU

        box_mask = np.ones(len(box_list))

        for i in range(len(box_list)):
            if box_mask[i] == 0: continue
            for j in range(i + 1, len(box_list)):
                if self._get_intersection_over_union(box_list[i], box_list[j]) > max_iou:
                    box_mask[j] = 0.0

        filter_iou_mask = np.array(box_mask > 0.0, dtype='bool')
        return filter_iou_mask
        

    # Converts the boxes in box list to pixel units
    # assumes box_list is the output from the box output from
    # the tiny yolo network and is [grid_size x grid_size x 2 x 4].
    def _boxes_to_pixel_units(self, box_list, image_width, image_height, grid_size):

        # number of boxes per grid cell
        boxes_per_cell = 2

        # setup some offset values to map boxes to pixels
        # box_offset will be [[ [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]] ...repeated for 7 ]
        box_offset = np.transpose(np.reshape(np.array([np.arange(grid_size)]*(grid_size*2)),(boxes_per_cell,grid_size, grid_size)),(1,2,0))

        # adjust the box center
        box_list[:,:,:,0] += box_offset
        box_list[:,:,:,1] += np.transpose(box_offset,(1, 0, 2))
        box_list[:,:,:,0:2] = box_list[:,:,:,0:2] / (grid_size * 1.0)

        # adjust the lengths and widths
        box_list[:,:,:,2] = np.multiply(box_list[:,:,:,2], box_list[:,:,:,2])
        box_list[:,:,:,3] = np.multiply(box_list[:,:,:,3], box_list[:,:,:,3])

        #scale the boxes to the image size in pixels
        box_list[:,:,:,0] *= image_width
        box_list[:,:,:,1] *= image_height
        box_list[:,:,:,2] *= image_width
        box_list[:,:,:,3] *= image_height


    # Evaluate the intersection-over-union for two boxes
    # The intersection-over-union metric determines how close
    # two boxes are to being the same box.  The closer the boxes
    # are to being the same, the closer the metric will be to 1.0
    # box_1 and box_2 are arrays of 4 numbers which are the (x, y)
    # points that define the center of the box and the length and width of
    # the box.
    # Returns the intersection-over-union (between 0.0 and 1.0)
    # for the two boxes specified.
    def _get_intersection_over_union(self, box_1, box_2):

        # one diminsion of the intersecting box
        intersection_dim_1 = min(box_1[0]+0.5*box_1[2],box_2[0]+0.5*box_2[2])-\
                             max(box_1[0]-0.5*box_1[2],box_2[0]-0.5*box_2[2])

        # the other dimension of the intersecting box
        intersection_dim_2 = min(box_1[1]+0.5*box_1[3],box_2[1]+0.5*box_2[3])-\
                             max(box_1[1]-0.5*box_1[3],box_2[1]-0.5*box_2[3])

        if intersection_dim_1 < 0 or intersection_dim_2 < 0 :
            # no intersection area
            intersection_area = 0
        else :
            # intersection area is product of intersection dimensions
            intersection_area =  intersection_dim_1*intersection_dim_2

        # calculate the union area which is the area of each box added
        # and then we need to subtract out the intersection area since
        # it is counted twice (by definition it is in each box)
        union_area = box_1[2]*box_1[3] + box_2[2]*box_2[3] - intersection_area;

        # now we can return the intersection over union
        iou = intersection_area / union_area

        return iou

