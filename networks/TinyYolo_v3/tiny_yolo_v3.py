
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

anchors = [10,14, 23,27, 37,58, 81,82, 135,169, 344,319]


def parse_args():
    parser = argparse.ArgumentParser(description = 'Image classifier using \
                         Intel® Neural Compute Stick 2.' )
    parser.add_argument( '--ir', metavar = 'IR_File',
                        type=str, default = 'tiny_yolo_v3.xml', 
                        help = 'Absolute path to the neural network IR xml file.')
    parser.add_argument( '-l', '--labels', metavar = 'LABEL_FILE', 
                        type=str, default = 'coco.names',
                        help='Absolute path to labels file.')
    parser.add_argument( '-i', '--image', metavar = 'IMAGE_FILE', 
                        type=str, default = '../../data/images/nps_chair.png',
                        help = 'Absolute path to image file.')
    parser.add_argument( '--threshold', metavar = 'FLOAT', 
                        type=float, default = 0.10,
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
    max_iou = 0.25

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



def display_info(input_shape, output_shape, image, ir, labels):
    print()
    print(YELLOW + 'Tiny Yolo v1: Starting application...' + NOCOLOR)
    print('   - ' + YELLOW + 'Plugin:       ' + NOCOLOR + 'Myriad')
    print('   - ' + YELLOW + 'IR File:     ' + NOCOLOR, ir)
    print('   - ' + YELLOW + 'Input Shape: ' + NOCOLOR, input_shape)
    print('   - ' + YELLOW + 'Output Shape:' + NOCOLOR, output_shape)
    print('   - ' + YELLOW + 'Labels File: ' + NOCOLOR, labels)
    print('   - ' + YELLOW + 'Image File:   ' + NOCOLOR, image)
    


def parseTinyYoloV3Output(output, objects, source_image_width, source_image_height, scaled_w, scaled_h):
    
    output = output.transpose(0,2,3,1)
    output_h = output.shape[1]
    output_w = output.shape[2]
    #print("output 0,0: ", output[0][0][0].size)
    # 80 class scores + 4 coordinate values + 1 objectness score = 85 values
    # 85 values * 3 prior box scores = 255 values 
    # 255 * either 26 or 13 scale 
    classes = 80
    num_anchor_boxes_per_cell = 3
    
    anchor_offset = 0
    if output_w == 13:
        anchor_offset = 2 * 3
    elif output_w == 26:
        anchor_offset = 2 * 0

    x_ratio = float(source_image_width) / scaled_w
    y_ratio = float(source_image_height) / scaled_h

    output_size = output_w * output_h
    for result_counter in range(output_size):
        row = int(result_counter / output_w)
        col = int(result_counter % output_h)
        for anchor_boxes in range(num_anchor_boxes_per_cell):
            x = (col + output[0][row][col][anchor_boxes * 85 + 0]) / output_w * scaled_w
            y = (row + output[0][row][col][anchor_boxes * 85 + 1]) / output_h * scaled_h
            width = np.exp(output[0][row][col][anchor_boxes * 85 + 2]) * anchors[anchor_offset + 2 * anchor_boxes]
            height = np.exp(output[0][row][col][anchor_boxes * 85 + 3]) * anchors[anchor_offset + 2 * anchor_boxes + 1]

            obj_score = output[0][row][col][anchor_boxes * 85 + 4]
            #print("objectness score: " + str(obj_score))
            if obj_score < 0.1:
                continue
            #print("\ngrid " + str(row) + " " + str(col) + " - anchor box: " + str(anchor_boxes))
            #print("output xmin: " + str(x))
            #print("output ymin: " + str(y))
            #print("output width: " + str(width))
            #print("output height: " + str(height))
            #print("objectness score: " + str(output[0][row][col][anchor_boxes * 85 + 4]))
            
            for class_id in range(classes):
                #print("class_index: ", result_counter)
                class_score = output[0][row][col][anchor_boxes * 85 + 5 + class_id]
                
                if (class_score * obj_score) < 0.25:
                    continue
                xmin = max(int((x - width / 2) * x_ratio), 0)
                ymin = max(int((y - height / 2) * y_ratio), 0)
                xmax = min(int(x + width * x_ratio), source_image_width-1)
                ymax = min(int(y + height * y_ratio), source_image_height-1)
                objects.append((xmin, ymin, xmax, ymax, obj_score*class_score, class_id))
                    #objects.append(ymin)
                    #objects.append(xmin)
                    #objects.append(ymax)
                    #objects.append(obj_score*class_score)
                    #objects.append(class_id)
    


# This function is called from the entry point to do
# all the work.
def main():
    ARGS = parse_args().parse_args()
    input_stream = ARGS.image
    labels = ARGS.labels
    if ARGS.image.lower() == "cam":
        input_stream = 0
    ir = ARGS.ir
    threshold = ARGS.threshold
    
    # Prepare Categories
    with open(labels) as labels_file:
	    label_list = labels_file.read().splitlines()
    
	    
    print(YELLOW + 'Running NCS Tensorflow TinyYolo v3 example...')

    ####################### 1. Setup Plugin and Network #######################
    # Select the myriad plugin and IRs to be used
    ie = IECore()
    net = IENetwork(model = ir, weights = ir[:-3] + 'bin')

    # Set up the input and output blobs
    input_blob = next(iter(net.inputs))
    output_blob = next(iter(net.outputs))
    input_shape = net.inputs[input_blob].shape
    output_shape = net.outputs[output_blob].shape
    
    # Display model information
    display_info(input_shape, output_shape, input_stream, ir, labels)
    
    # Load the network and get the network shape information
    exec_net = ie.load_network(network = net, device_name = DEVICE)
    n, c, h, w = input_shape
    
    cap = cv2.VideoCapture(input_stream)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    ret, input_image = cap.read()

    #print("output h:", output_h)
    #print("output w:", output_w)
    while cap.isOpened():
        # Read image from file, resize it to network width and height

        display_image = input_image
        input_image = cv2.resize(input_image, (w, h), cv2.INTER_LINEAR)
        input_image = input_image.astype(np.float32)
        input_image = np.transpose(input_image, (2,0,1))
        reshaped_image = input_image.reshape((n, c, h, w))
        
        # Perform the inference asynchronously
        req_handle = exec_net.start_async(request_id=0, inputs={input_blob: reshaped_image})
        status = req_handle.wait()
        output = req_handle.outputs
        
        # Width and height calculations
        source_image_width = display_image.shape[1]
        source_image_height = display_image.shape[0]
        scaled_w = int(source_image_width * min(w/source_image_width, w/source_image_height))
        scaled_h = int(source_image_height * min(h/source_image_width, h/source_image_height))
        
        # Post-processing for tiny yolo
        objects = []
        for output_result in output.values():
            parseTinyYoloV3Output(output_result, objects, source_image_width, source_image_height, scaled_w, scaled_h)
        
        filtered_mask = get_duplicate_box_mask(objects)
        
        for num in range((len(objects))):
            if filtered_mask[num] == True:
                label_background_color = (70, 120, 70) # greyish green background for text
                label_text_color = (255, 255, 255)   # white text
                cv2.rectangle(display_image,(objects[num][0], objects[num][1]), (objects[num][2], objects[num][1]+20), label_background_color, -1)
                cv2.putText(display_image, label_list[objects[num][5]] + ' : %.2f' % objects[num][4], (objects[num][0]+5, objects[num][1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)
                cv2.rectangle(display_image, (objects[num][0], objects[num][1]), (objects[num][2], objects[num][3]), (0,255,0), 1)

            

        #filtered_objs = filter_objects(output.astype(np.float32), input_image.shape[1], input_image.shape[2], label_list, threshold)
        cv2.imshow("tiny yolo v3", display_image)
        # get another frame from camera if using camera input
        if input_stream == 0:
            key = cv2.waitKey(1)
            if key != -1:
                cap.release()
                break
            ret, frame = cap.read()
        else: #  or wait for key press if image input
            while True:
                key = cv2.waitKey(1)
                if key != -1:
                    cap.release()
                    break

    print('\n Displaying image with objects detected in GUI...')
    print(' Click in the GUI window and hit any key to exit.')
    # display the filtered objects/boxes in a GUI window
    #display_objects_in_gui(display_image, filtered_objs, input_image.shape[1], input_image.shape[2])
    cv2.destroyAllWindows()
    print('\n Finished.')
    del net
    del exec_net


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())
