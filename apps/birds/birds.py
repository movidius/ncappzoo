#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.
# NPS

from mvnc import mvncapi as mvnc
import sys
import numpy as np
import cv2
import os

# will execute on all images in this directory
input_image_path = './images'

tiny_yolo_graph_file= './yolo_tiny.graph'
googlenet_graph_file= './googlenet.graph'

# Tiny Yolo assumes input images are these dimensions.
TY_NETWORK_IMAGE_WIDTH = 448
TY_NETWORK_IMAGE_HEIGHT = 448

# GoogLeNet assumes input images are these dimensions
GN_NETWORK_IMAGE_WIDTH = 224
GN_NETWORK_IMAGE_HEIGHT = 224

# googlenet mean values will be read in from .npy file
gn_mean = [0., 0., 0.]

# labels to display along with boxes if googlenet classification is good
gn_labels = [""]

cv_window_name = 'Birds - Q to quit or any key to advance'


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
def filter_objects(inference_result, input_image_width, input_image_height):

    # the raw number of floats returned from the inference (GetResult())
    num_inference_results = len(inference_result)

    # the 20 classes this network was trained on
    network_classifications = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                               "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                               "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

    # birds only
    network_classifications_mask = [0, 0, 1, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0,0]


    # only keep boxes with probabilities greater than this
    probability_threshold = 0.10 # 0.07

    num_classifications = len(network_classifications) # should be 20
    grid_size = 7 # the image is a 7x7 grid.  Each box in the grid is 64x64 pixels
    boxes_per_grid_cell = 2 # the number of boxes returned for each grid cell

    # grid_size is 7 (grid is 7x7)
    # num classifications is 20
    # boxes per grid cell is 2
    all_probabilities = np.zeros((grid_size, grid_size, boxes_per_grid_cell, num_classifications))

    # classification_probabilities  contains a probability for each classification for
    # each 64x64 pixel square of the grid.  The source image contains
    # 7x7 of these 64x64 pixel squares and there are 20 possible classifications
    classification_probabilities = \
        np.reshape(inference_result[0:980], (grid_size, grid_size, num_classifications))
    num_of_class_probs = len(classification_probabilities)

    # The probability scale factor for each box
    box_prob_scale_factor = np.reshape(inference_result[980:1078], (grid_size, grid_size, boxes_per_grid_cell))

    # get the boxes from the results and adjust to be pixel units
    all_boxes = np.reshape(inference_result[1078:], (grid_size, grid_size, boxes_per_grid_cell, 4))
    boxes_to_pixel_units(all_boxes, input_image_width, input_image_height, grid_size)

    # adjust the probabilities with the scaling factor
    for box_index in range(boxes_per_grid_cell): # loop over boxes
        for class_index in range(num_classifications): # loop over classifications
            all_probabilities[:,:,box_index,class_index] = np.multiply(classification_probabilities[:,:,class_index],box_prob_scale_factor[:,:,box_index])


    probability_threshold_mask = np.array(all_probabilities>=probability_threshold, dtype='bool')
    box_threshold_mask = np.nonzero(probability_threshold_mask)
    boxes_above_threshold = all_boxes[box_threshold_mask[0],box_threshold_mask[1],box_threshold_mask[2]]
    classifications_for_boxes_above = np.argmax(all_probabilities,axis=3)[box_threshold_mask[0],box_threshold_mask[1],box_threshold_mask[2]]
    probabilities_above_threshold = all_probabilities[probability_threshold_mask]

    # sort the boxes from highest probability to lowest and then
    # sort the probabilities and classifications to match
    argsort = np.array(np.argsort(probabilities_above_threshold))[::-1]
    boxes_above_threshold = boxes_above_threshold[argsort]
    classifications_for_boxes_above = classifications_for_boxes_above[argsort]
    probabilities_above_threshold = probabilities_above_threshold[argsort]


    # get mask for boxes that seem to be the same object
    duplicate_box_mask = get_duplicate_box_mask(boxes_above_threshold)

    # update the boxes, probabilities and classifications removing duplicates.
    boxes_above_threshold = boxes_above_threshold[duplicate_box_mask]
    classifications_for_boxes_above = classifications_for_boxes_above[duplicate_box_mask]
    probabilities_above_threshold = probabilities_above_threshold[duplicate_box_mask]

    classes_boxes_and_probs = []
    for i in range(len(boxes_above_threshold)):
        if (network_classifications_mask[classifications_for_boxes_above[i]] != 0):
            classes_boxes_and_probs.append([network_classifications[classifications_for_boxes_above[i]],boxes_above_threshold[i][0],boxes_above_threshold[i][1],boxes_above_threshold[i][2],boxes_above_threshold[i][3],probabilities_above_threshold[i]])

    return classes_boxes_and_probs

# creates a mask to remove duplicate objects (boxes) and their related probabilities and classifications
# that should be considered the same object.  This is determined by how similar the boxes are
# based on the intersection-over-union metric.
# box_list is as list of boxes (4 floats for centerX, centerY and Length and Width)
def get_duplicate_box_mask(box_list):
    # The intersection-over-union threshold to use when determining duplicates.
    # objects/boxes found that are over this threshold will be
    # considered the same object
    max_iou = 0.25

    box_mask = np.ones(len(box_list))

    for i in range(len(box_list)):
        if box_mask[i] == 0: continue
        for j in range(i + 1, len(box_list)):
            if get_intersection_over_union(box_list[i], box_list[j]) > max_iou:
                box_mask[j] = 0.0

    filter_iou_mask = np.array(box_mask > 0.0, dtype='bool')
    return filter_iou_mask

# Converts the boxes in box list to pixel units
# assumes box_list is the output from the box output from
# the tiny yolo network and is [grid_size x grid_size x 2 x 4].
def boxes_to_pixel_units(box_list, image_width, image_height, grid_size):

    # number of boxes per grid cell
    boxes_per_cell = 2

    # setup some offset values to map boxes to pixels
    # box_offset will be [[ [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]] ...repeated for 7 ]
    box_offset = np.transpose(np.reshape(np.array([np.arange(grid_size)]*(grid_size*2)),(boxes_per_cell,grid_size, grid_size)),(1,2,0))

    # adjust the box center
    box_list[:,:,:,0] += box_offset
    box_list[:,:,:,1] += np.transpose(box_offset,(1,0,2))
    box_list[:,:,:,0:2] = box_list[:,:,:,0:2] / (grid_size * 1.0)

    # adjust the lengths and widths
    box_list[:,:,:,2] = np.multiply(box_list[:,:,:,2],box_list[:,:,:,2])
    box_list[:,:,:,3] = np.multiply(box_list[:,:,:,3],box_list[:,:,:,3])

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
def get_intersection_over_union(box_1, box_2):

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


# Displays a gui window with an image that contains
# boxes and lables for found objects.  will not return until
# user presses a key or times out.
# source_image is the original image before resizing or otherwise changed
#
# filtered_objects is a list of lists (as returned from filter_objects()
#   and then added to by get_googlenet_classifications()
#   each of the inner lists represent one found object and contain
#   the following values:
#     string that is yolo network classification ie 'bird'
#     float value for box center X pixel location within source image
#     float value for box center Y pixel location within source image
#     float value for box width in pixels within source image
#     float value for box height in pixels within source image
#     float value that is the probability for the yolo classification.
#     int value that is the index of the googlenet classification
#     string value that is the googlenet classification string.
#     float value that is the googlenet probability
#
# Returns true if should go to next image or false if
# should not.
def display_objects_in_gui(source_image, filtered_objects):

    DISPLAY_BOX_WIDTH_PAD = 0
    DISPLAY_BOX_HEIGHT_PAD = 20

    # if googlenet returns a probablity less than this then
    # just use the tiny yolo more general classification ie 'bird'
    GOOGLE_PROB_MIN = 0.5

	# copy image so we can draw on it.
    display_image = source_image.copy()
    source_image_width = source_image.shape[1]
    source_image_height = source_image.shape[0]

    x_ratio = float(source_image_width) / TY_NETWORK_IMAGE_WIDTH
    y_ratio = float(source_image_height) / TY_NETWORK_IMAGE_HEIGHT

    # loop through each box and draw it on the image along with a classification label
    for obj_index in range(len(filtered_objects)):
        center_x = int(filtered_objects[obj_index][1] * x_ratio)
        center_y = int(filtered_objects[obj_index][2]* y_ratio)
        half_width = int(filtered_objects[obj_index][3]*x_ratio)//2 + DISPLAY_BOX_WIDTH_PAD
        half_height = int(filtered_objects[obj_index][4]*y_ratio)//2 + DISPLAY_BOX_HEIGHT_PAD

        # calculate box (left, top) and (right, bottom) coordinates
        box_left = max(center_x - half_width, 0)
        box_top = max(center_y - half_height, 0)
        box_right = min(center_x + half_width, source_image_width)
        box_bottom = min(center_y + half_height, source_image_height)

        #draw the rectangle on the image.  This is hopefully around the object
        box_color = (0, 255, 0)  # green box
        box_thickness = 2
        cv2.rectangle(display_image, (box_left, box_top),(box_right, box_bottom), box_color, box_thickness)

        # draw the classification label string just above and to the left of the rectangle
        label_background_color = (70, 120, 70) # greyish green background for text
        label_text_color = (255, 255, 255)   # white text

        if (filtered_objects[obj_index][8] > GOOGLE_PROB_MIN):
            label_text = filtered_objects[obj_index][7] + ' : %.2f' % filtered_objects[obj_index][8]
        else:
            label_text = filtered_objects[obj_index][0] + ' : %.2f' % filtered_objects[obj_index][5]

        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_left = box_left
        label_top = box_top - label_size[1]
        label_right = label_left + label_size[0]
        label_bottom = label_top + label_size[1]
        cv2.rectangle(display_image,(label_left-1, label_top-1),(label_right+1, label_bottom+1), label_background_color, -1)

        # label text above the box
        cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

    # display text to let user know how to quit
    cv2.rectangle(display_image,(0, 0),(140, 30), (128, 128, 128), -1)
    cv2.putText(display_image, "Q to Quit", (10, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    cv2.putText(display_image, "Any key to advance", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    cv2.imshow(cv_window_name, display_image)
    raw_key = cv2.waitKey(3000)
    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
        return False

    return True



# Executes googlenet inferences on all objects defined by filtered_objects
# To run the inferences will crop an image out of source image based on the
# boxes defined in filtered_objects and use that as input for googlenet.
#
# gn_graph is the googlenet graph object on which the inference should be executed.
#
# source_image the original image on which the inference was run.  The boxes
#   defined by filtered_objects are rectangles within this image and will be
#   used as input for googlenet.  This image may be scaled differently from 
#   the tiny yolo network dimensions in which case the boxes in filtered objects
#   will be scaled to match.
#
# filtered_objects [IN/OUT] upon input is a list of lists (as returned from filter_objects()
#   each of the inner lists represent one found object and contain
#   the following 6 values:
#     string that is network classification ie 'cat', or 'chair' etc
#     float value for box center X pixel location within source image
#     float value for box center Y pixel location within source image
#     float value for box width in pixels within source image
#     float value for box height in pixels within source image
#     float value that is the probability for the network classification.
#   upon output the following 3 values from the googlenet inference will
#   be added to each inner list of filtered_objects
#     int value that is the index of the googlenet classification
#     string value that is the googlenet classification string.
#     float value that is the googlenet probability
#
# returns None
def get_googlenet_classifications(gn_graph, source_image, filtered_objects):

    source_image_width = source_image.shape[1]
    source_image_height = source_image.shape[0]
    x_scale = float(source_image_width) / TY_NETWORK_IMAGE_WIDTH
    y_scale = float(source_image_height) / TY_NETWORK_IMAGE_HEIGHT

    # pad the height and width of the image boxes by this amount
    # to make sure we get the whole object in the image that
    # we pass to googlenet
    WIDTH_PAD = int(20 * x_scale)
    HEIGHT_PAD = int(30 * y_scale)

    # loop through each box and crop the image in that rectangle
    # from the source image and then use it as input for googlenet
    for obj_index in range(len(filtered_objects)):
        center_x = int(filtered_objects[obj_index][1]*x_scale)
        center_y = int(filtered_objects[obj_index][2]*y_scale)
        half_width = int(filtered_objects[obj_index][3]*x_scale)//2 + WIDTH_PAD
        half_height = int(filtered_objects[obj_index][4]*y_scale)//2 + HEIGHT_PAD

        # calculate box (left, top) and (right, bottom) coordinates
        box_left = max(center_x - half_width, 0)
        box_top = max(center_y - half_height, 0)
        box_right = min(center_x + half_width, source_image_width)
        box_bottom = min(center_y + half_height, source_image_height)

        one_image = source_image[box_top:box_bottom, box_left:box_right]
        filtered_objects[obj_index] += googlenet_inference(gn_graph, one_image)
        #print (googlenet_inference(gn_graph, one_image))

    return


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
def googlenet_inference(gn_graph, input_image):

    # Resize image to googlenet network width and height
    # then convert to float32, normalize (divide by 255),
    # and finally convert to convert to float16 to pass to LoadTensor as input for an inference
    input_image = cv2.resize(input_image, (GN_NETWORK_IMAGE_WIDTH, GN_NETWORK_IMAGE_HEIGHT), cv2.INTER_LINEAR)
    input_image = input_image.astype(np.float32)
    input_image[:, :, 0] = (input_image[:, :, 0] - gn_mean[0])
    input_image[:, :, 1] = (input_image[:, :, 1] - gn_mean[1])
    input_image[:, :, 2] = (input_image[:, :, 2] - gn_mean[2])

    # Load tensor and get result.  This executes the inference on the NCS
    gn_graph.LoadTensor(input_image.astype(np.float16), 'googlenet')
    output, userobj = gn_graph.GetResult()

    order = output.argsort()[::-1][:1]

    '''
    print('\n------- prediction --------')
    for i in range(0, 1):
        print('prediction ' + str(i) + ' (probability ' + str(output[order[i]]) + ') is ' + labels[
            order[i]] + '  label index is: ' + str(order[i]))
    '''

    # index, label, probability
    ret_index = order[0]
    ret_label = gn_labels[order[0]]
    ret_prob = output[order[0]]

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



# This function is called from the entry point to do
# all the work.
def main():
    global gn_mean, gn_labels, input_image_filename_list

    print('Running NCS birds example')

    # get list of all the .jpg files in the image directory
    input_image_filename_list = os.listdir(input_image_path)
    input_image_filename_list = [input_image_path + '/' + i for i in input_image_filename_list if i.endswith('.jpg')]

    if (len(input_image_filename_list) < 1):
        # no images to show
        print('No .jpg files found')
        return 1

    # Set logging level and initialize/open the first NCS we find
    mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 0)
    devices = mvnc.EnumerateDevices()
    if len(devices) < 2:
        print('This application requires two NCS devices.')
        print('Insert two devices and try again!')
        return 1
    ty_device = mvnc.Device(devices[0])
    ty_device.OpenDevice()

    gn_device = mvnc.Device(devices[1])
    gn_device.OpenDevice()


    #Load tiny yolo graph from disk and allocate graph via API
    with open(tiny_yolo_graph_file, mode='rb') as ty_file:
        ty_graph_from_disk = ty_file.read()
    ty_graph = ty_device.AllocateGraph(ty_graph_from_disk)

    #Load googlenet graph from disk and allocate graph via API
    with open(googlenet_graph_file, mode='rb') as gn_file:
        gn_graph_from_disk = gn_file.read()
    gn_graph = gn_device.AllocateGraph(gn_graph_from_disk)

    # GoogLenet initialization
    EXAMPLES_BASE_DIR = '../../'
    gn_mean = np.load(EXAMPLES_BASE_DIR + 'data/ilsvrc12/ilsvrc_2012_mean.npy').mean(1).mean(1)  # loading the mean file

    gn_labels_file = EXAMPLES_BASE_DIR + 'data/ilsvrc12/synset_words.txt'
    gn_labels = np.loadtxt(gn_labels_file, str, delimiter='\t')
    for label_index in range(0, len(gn_labels)):
        temp = gn_labels[label_index].split(',')[0].split(' ', 1)[1]
        gn_labels[label_index] = temp


    print('Q to quit, or any key to advance to next image')

    cv2.namedWindow(cv_window_name)

    for input_image_file in input_image_filename_list :
        # Read image from file, resize it to network width and height
        # save a copy in img_cv for display, then convert to float32, normalize (divide by 255),
        # and finally convert to convert to float16 to pass to LoadTensor as input for an inference
        input_image = cv2.imread(input_image_file)

        # resize the image to be a standard width for all images and maintain aspect ratio
        STANDARD_RESIZE_WIDTH = 800
        input_image_width = input_image.shape[1]
        input_image_height = input_image.shape[0]
        standard_scale = float(STANDARD_RESIZE_WIDTH) / input_image_width
        new_width = int(input_image_width * standard_scale) # this should be == STANDARD_RESIZE_WIDTH
        new_height = int(input_image_height * standard_scale)
        input_image = cv2.resize(input_image, (new_width, new_height), cv2.INTER_LINEAR)

        display_image = input_image
        input_image = cv2.resize(input_image, (TY_NETWORK_IMAGE_WIDTH, TY_NETWORK_IMAGE_HEIGHT), cv2.INTER_LINEAR)
        input_image = input_image[:, :, ::-1]  # convert to RGB
        input_image = input_image.astype(np.float32)
        input_image = np.divide(input_image, 255.0)

        # Load tensor and get result.  This executes the inference on the NCS
        ty_graph.LoadTensor(input_image.astype(np.float16), 'user object')
        output, userobj = ty_graph.GetResult()

        # filter out all the objects/boxes that don't meet thresholds
        filtered_objs = filter_objects(output.astype(np.float32), input_image.shape[1], input_image.shape[0]) # fc27 instead of fc12 for yolo_small

        get_googlenet_classifications(gn_graph, display_image, filtered_objs)

        # check if the window has been closed.  all properties will return -1.0
        # for windows that are closed. If the user has closed the window via the
        # x on the title bar then we will break out of the loop.  we are
        # getting property aspect ratio but it could probably be any property
        try:
            prop_asp = cv2.getWindowProperty(cv_window_name, cv2.WND_PROP_ASPECT_RATIO)
        except:
            break;
        prop_asp = cv2.getWindowProperty(cv_window_name, cv2.WND_PROP_ASPECT_RATIO)
        if (prop_asp < 0.0):
            # the property returned was < 0 so assume window was closed by user
            break;

        ret_val = display_objects_in_gui(display_image, filtered_objs)
        if (not ret_val):
            break


    # clean up tiny yolo
    ty_graph.DeallocateGraph()
    ty_device.CloseDevice()

    # Clean up googlenet
    gn_graph.DeallocateGraph()
    gn_device.CloseDevice()

    print('Finished')


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())
