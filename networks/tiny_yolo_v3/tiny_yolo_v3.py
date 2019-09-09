
# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.


GREEN = '\033[1;32m'
RED = '\033[1;31m'
NOCOLOR = '\033[0m'
YELLOW = '\033[1;33m'
DEVICE = "MYRIAD"

try:
    from openvino.inference_engine import IENetwork, IECore
except:
    print(RED + '\nPlease make sure your OpenVINO environment variables are set by sourcing the' + YELLOW + ' setupvars.sh ' + RED + 'script found in <your OpenVINO install location>/bin/ folder.\n' + NOCOLOR)
    exit(1)

import sys
import numpy as np
import cv2
import argparse

# Adjust these thresholds
DETECTION_THRESHOLD = 0.60
IOU_THRESHOLD = 0.25

# Tiny yolo anchor box values
anchors = [10,14, 23,27, 37,58, 81,82, 135,169, 344,319]

# Used for display
BOX_COLOR = (0,255,0)
LABEL_BG_COLOR = (70, 120, 70) # greyish green background for text
TEXT_COLOR = (255, 255, 255)   # white text
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
WINDOW_SIZE_W = 640
WINDOW_SIZE_H = 480


# Parses arguments for the application
def parse_args():
    parser = argparse.ArgumentParser(description = 'Image classifier using \
                         Intel® Neural Compute Stick 2.' )
    parser.add_argument( '--ir', metavar = 'IR_File',
                        type=str, default = 'tiny_yolo_v3.xml', 
                        help = 'Absolute path to the neural network IR xml file.')
    parser.add_argument( '-l', '--labels', metavar = 'LABEL_FILE', 
                        type=str, default = 'coco.names',
                        help='Absolute path to labels file.')
    parser.add_argument( '-i', '--input', metavar = 'IMAGE_FILE or cam', 
                        type=str, default = '../../data/images/nps_chair.png',
                        help = 'Absolute path to image file or cam for camera stream.')
    parser.add_argument( '--threshold', metavar = 'FLOAT', 
                        type=float, default = DETECTION_THRESHOLD,
                        help = 'Threshold for detection.')
                      
    return parser



# creates a mask to remove duplicate objects (boxes) and their related probabilities and classifications
# that should be considered the same object.  This is determined by how similar the boxes are
# based on the intersection-over-union metric.
# box_list is as list of boxes (4 floats for centerX, centerY and Length and Width)
def get_duplicate_box_mask(box_list):
    # The intersection-over-union threshold to use when determining duplicates.
    # objects/boxes found that are over this threshold will be
    # considered the same object
    max_iou = IOU_THRESHOLD

    box_mask = np.ones(len(box_list))

    for i in range(len(box_list)):
        if box_mask[i] == 0: continue
        for j in range(i + 1, len(box_list)):
            if get_intersection_over_union(box_list[i], box_list[j]) >= max_iou:
                if box_list[i][4] < box_list[j][4]:
                    box_list[i], box_list[j] = box_list[j], box_list[i]
                box_mask[j] = 0.0

    filter_iou_mask = np.array(box_mask > 0.0, dtype='bool')
    return filter_iou_mask
    

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
    #print("iou: ", iou)
    return iou


# displays basic information regarding the model
def display_info(input_shape, net_outputs, image, ir, labels):
    
    output_nodes = []
    output_iter = iter(net_outputs)
    for i in range(len(net_outputs)):
        output_nodes.append(next(output_iter))

    print()
    print(YELLOW + 'Tiny Yolo v3: Starting application...' + NOCOLOR)
    print('   - ' + YELLOW + 'Plugin:       ' + NOCOLOR + 'Myriad')
    print('   - ' + YELLOW + 'IR File:     ' + NOCOLOR, ir)
    print('   - ' + YELLOW + 'Input Shape: ' + NOCOLOR, input_shape)
    print('   - ' + YELLOW + 'Output Shapes:' + NOCOLOR)
    for j in range(len(output_nodes)):
        print('      - '+YELLOW+'output #' + str(j) + ' name: ' + NOCOLOR + output_nodes[j])
        print('         - output shape: ' + NOCOLOR + str(net_outputs[output_nodes[j]].shape))
    print('   - ' + YELLOW + 'Labels File: ' + NOCOLOR, labels)
    print('   - ' + YELLOW + 'Image File:   ' + NOCOLOR, image)
    

# This function parses the output results from tiny yolo v3.
# The results are transposed so the output shape is (1, 13, 13, 255) or (1, 26, 26, 255). Original will be (1, 255, w, h).
# Tiny yolo does detection on two different scales using 13x13 grid and 26x26 grid.
# This is how the output is parsed:
# Imagine the image being split up into 13x13 or 26x26 grid. Each grid cell contains 3 anchor boxes. 
# For each of those 3 anchor boxes, there are 85 values. 
# 80 class probabilities + 4 coordinate values + 1 box confidence score = 85 values 
# So that results in each grid cell having 255 values (85 values x 3 anchor boxes = 255 values)
def parseTinyYoloV3Output(output_node_results, filtered_objects, source_image_width, source_image_height, scaled_w, scaled_h, detection_threshold, num_labels):
    # transpose the output node results
    output_node_results = output_node_results.transpose(0,2,3,1)
    output_h = output_node_results.shape[1]
    output_w = output_node_results.shape[2]

    # 80 class scores + 4 coordinate values + 1 objectness score = 85 values
    # 85 values * 3 prior box scores per grid cell= 255 values 
    # 255 values * either 26 or 13 grid cells
    num_of_classes = num_labels
    num_anchor_boxes_per_cell = 3
    
    # Set the anchor offset depending on the output result shape
    anchor_offset = 0
    if output_w == 13:
        anchor_offset = 2 * 3
    elif output_w == 26:
        anchor_offset = 2 * 0

	# used to calculate approximate coordinates of bounding box
    x_ratio = float(source_image_width) / scaled_w
    y_ratio = float(source_image_height) / scaled_h

	# Filter out low scoring results
    output_size = output_w * output_h
    for result_counter in range(output_size): 
        row = int(result_counter / output_w)
        col = int(result_counter % output_h)
        for anchor_boxes in range(num_anchor_boxes_per_cell): 
        	# check the box confidence score of the anchor box. This is how likely the box contains an object
            box_confidence_score = output_node_results[0][row][col][anchor_boxes * num_of_classes + 5 + 4]
            if box_confidence_score < detection_threshold:
                continue
            # Calculate the x, y, width, and height of the box
            x_center = (col + output_node_results[0][row][col][anchor_boxes * num_of_classes + 5 + 0]) / output_w * scaled_w
            y_center = (row + output_node_results[0][row][col][anchor_boxes * num_of_classes + 5 + 1]) / output_h * scaled_h
            width = np.exp(output_node_results[0][row][col][anchor_boxes * num_of_classes + 5 + 2]) * anchors[anchor_offset + 2 * anchor_boxes]
            height = np.exp(output_node_results[0][row][col][anchor_boxes * num_of_classes + 5 + 3]) * anchors[anchor_offset + 2 * anchor_boxes + 1]
            # Now we check for anchor box for the highest class probabilities.
            # If the probability exceeds the threshold, we save the box coordinates, class score and class id
            for class_id in range(num_of_classes): 
                class_probability = output_node_results[0][row][col][anchor_boxes * num_of_classes + 5 + 5 + class_id]
                # Calculate the class's confidence score by multiplying the box_confidence score by the class probabiity
                class_confidence_score = class_probability * box_confidence_score
                if (class_confidence_score) < detection_threshold:
                    continue
                # Calculate the bounding box top left and bottom right vertexes
                xmin = max(int((x_center - width / 2) * x_ratio), 0)
                ymin = max(int((y_center - height / 2) * y_ratio), 0)
                xmax = min(int(xmin + width * x_ratio), source_image_width-1)
                ymax = min(int(ymin + height * y_ratio), source_image_height-1)
                filtered_objects.append((xmin, ymin, xmax, ymax, class_confidence_score, class_id))
    


# This function is called from the entry point to do
# all the work.
def main():
	# Argument parsing and parameter setting
    ARGS = parse_args().parse_args()
    input_stream = ARGS.input
    labels = ARGS.labels
    if ARGS.input.lower() == "cam" or ARGS.input.lower() == "camera":
        input_stream = 0
    ir = ARGS.ir
    detection_threshold = ARGS.threshold
    
    # Prepare Categories
    with open(labels) as labels_file:
        label_list = labels_file.read().splitlines()
	    
    print(YELLOW + 'Running OpenVINO NCS Tensorflow TinyYolo v3 example...' + NOCOLOR)
    print('\n Displaying image with objects detected in GUI...')
    print(' Click in the GUI window and hit any key to exit.')

    ####################### 1. Create ie core and network #######################
    # Select the myriad plugin and IRs to be used
    ie = IECore()
    net = IENetwork(model = ir, weights = ir[:-3] + 'bin')

    # Set up the input blobs
    input_blob = next(iter(net.inputs))
    input_shape = net.inputs[input_blob].shape

    # Display model information
    display_info(input_shape, net.outputs, input_stream, ir, labels)
    
    # Load the network and get the network input shape information
    exec_net = ie.load_network(network = net, device_name = DEVICE)
    n, c, network_input_h, network_input_w = input_shape
    
    # Prepare the input stream
    cap = cv2.VideoCapture(input_stream)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_SIZE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_SIZE_H)
    
    # Read a frame
    ret, frame = cap.read()
    # Width and height calculations. These will be used to scale the bounding boxes
    source_image_width = frame.shape[1]
    source_image_height = frame.shape[0]
    scaled_w = int(source_image_width * min(network_input_w/source_image_width, network_input_w/source_image_height))
    scaled_h = int(source_image_height * min(network_input_h/source_image_width, network_input_h/source_image_height))

    while cap.isOpened():
        # Make a copy of the original frame. Get the frame's width and height.
        if frame is None:
            print(RED + "\nUnable to read the input." + NOCOLOR)
            quit()
	    ####################### 2. Preprocessing #######################
        # Image preprocessing
        frame = cv2.flip(frame, 1)

        display_image = frame

        # Image preprocessing (resize, transpose, reshape)
        input_image = cv2.resize(frame, (network_input_w, network_input_h), cv2.INTER_LINEAR)
        input_image = input_image.astype(np.float32)
        input_image = np.transpose(input_image, (2,0,1))
        reshaped_image = input_image.reshape((n, c, network_input_h, network_input_w))
	    ####################### 3. Perform Inference #######################
        # Perform the inference asynchronously
        req_handle = exec_net.start_async(request_id=0, inputs={input_blob: reshaped_image})
        status = req_handle.wait()
	    ####################### 4. Get results #######################
        all_output_results = req_handle.outputs
        
	    ####################### 5. Post processing for results #######################
        # Post-processing for tiny yolo v3 
        # The post process consists of the following steps:
        # 1. Parse the output and filter out low scores
        # 2. Filter out duplicates using intersection over union
        # 3. Draw boxes and text
        
        ## 1. Tiny yolo v3 has two outputs and we check/parse both outputs
        filtered_objects = []
        for output_node_results in all_output_results.values():
            parseTinyYoloV3Output(output_node_results, filtered_objects, source_image_width, source_image_height, scaled_w, scaled_h, detection_threshold, len(label_list))
        
        ## 2. Filter out duplicate objects from all detected objects
        filtered_mask = get_duplicate_box_mask(filtered_objects)
        ## 3. Draw rectangles and set up display texts
        for object_index in range(len(filtered_objects)):
            if filtered_mask[object_index] == True:
                # get all values from the filtered object list
                xmin = filtered_objects[object_index][0]
                ymin = filtered_objects[object_index][1]
                xmax = filtered_objects[object_index][2]
                ymax = filtered_objects[object_index][3]
                confidence = filtered_objects[object_index][4]
                class_id = filtered_objects[object_index][5]
                # Set up the text for display
                cv2.rectangle(display_image,(xmin, ymin), (xmax, ymin+20), LABEL_BG_COLOR, -1)
                cv2.putText(display_image, label_list[class_id] + ': %.2f' % confidence, (xmin+5, ymin+15), TEXT_FONT, 0.5, TEXT_COLOR, 1)
                # Set up the bounding box
                cv2.rectangle(display_image, (xmin, ymin), (xmax, ymax), BOX_COLOR, 1)
        
        # Now we can display the results!
        cv2.imshow("OpenVINO Tiny yolo v3 - Press any key to quit", display_image)
        
        # Handle key presses
        # Get another frame from camera if using camera input
        if input_stream == 0:
            key = cv2.waitKey(1)
            if key != -1:  # if used pressed a key, release the capture stream and break
                cap.release()
                break
            ret, frame = cap.read() # read another frame
        else: #  or wait for key press if image input
            print("\n Press any key to quit.")
            while True:
                key = cv2.waitKey(1)
                if key != -1:
                    cap.release()
                    break
                    
    # Clean up
    cv2.destroyAllWindows()
    del net
    del exec_net
    print('\n Finished.')
    

# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())
