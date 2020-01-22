from openvino.inference_engine import IENetwork, IECore
import numpy
import cv2
import sys
sys.path.append('../../shared/Python/')
from Object_detector import *

class Tiny_yolo_v3_object_detector(Object_detector):
    def __init__(self, ie,              # type: IECore 
                       net,             # type: IENetwork
                       device="MYRIAD"  # type: str
                ):
        # Get the input and output node names
        self.input_blob = next(iter(net.inputs))
        self.output_blob = next(iter(net.outputs))

       # Get the input and output shapes from the input/output nodes
        self.input_shape = net.inputs[self.input_blob].shape
        self.output_shape = net.outputs[self.output_blob].shape
        self.input_batchsize, self.input_channels, self.input_height, self.input_width = self.input_shape
        self.output_x, self.output_y, self.detections_count, self.detections_size = self.output_shape
        self.exec_net = ie.load_network(network = net, device_name = device)
        # Tiny yolo anchor box values
        self.anchors = [10,14, 23,27, 37,58, 81,82, 135,169, 344,319]
        self.iou_threshold = 0.25
        self.detection_threshold = 0.70
        self.num_of_classes = 80
        

    def __preprocess_image(self, src_image # type: OpenCV Mat
                          ):
        # type: (...) -> OpenCV Mat
        """ Perform image preprocessing: First resize the image, then transpose the image (HWC -> CHW). """
        image_to_preprocess = cv2.resize(src_image, (self.input_width, self.input_height))
        image_to_preprocess = numpy.transpose(image_to_preprocess, (2, 0, 1))
        preprocessed_img = image_to_preprocess.reshape((self.input_batchsize, self.input_channels, self.input_height, self.input_width))
        
        return preprocessed_img
        
        
    def run_inference_sync(self, original_image # type: OpenCV Mat
                          ):
        # type: (...) -> List
        """ Run an inference on the NCS, then perform postprocessing on the results and return them. """
        # send the image for preprocessing """
        preprocessed_image = self.__preprocess_image(original_image)
        cur_request_id = 0

        request_handle = self.exec_net.start_async(request_id=cur_request_id, inputs={self.input_blob: preprocessed_image})
        
        # wait for inference to complete
        if self.exec_net.requests[cur_request_id].wait(-1) == 0:
            detection_results_to_return = self.__postprocess(request_handle, original_image)
            return detection_results_to_return

    
    def __postprocess(self, request_handle, # type: OpenVINO InferRequest 
                            original_image  # type: OpenCV Mat
                     ):
        # type: (...) -> List
        """ Organizes the network inference results and then returns the results as a List. 
            The results will be returned as a list of tuples in the following format: 
            
            (Left side of bbox, Top side of bbox, Right side of bbox, Bottom of the bbox, Confidence score, Class id)
            
            Pseudocode Example:
                all_results = do_inference()
                results_to_return = []
                
                for result in all_results:
                    results_to_return.append((result.left_box, result.top_box, result.right_box, result.bottom_box, result.confidence_score, result.class_id))
            
                return results_to_return """

        # get the inference result
        all_output_results = request_handle.outputs
                
        ## Tiny yolo v3 has two outputs and we check/parse both outputs
        filtered_objects = []
        for output_node_results in all_output_results.values():
            print type(output_node_results)
            self.__parseTinyYoloV3Output(output_node_results, filtered_objects, original_image)
        
        ## Filter out duplicate objects from all detected objects
        filtered_mask = self.__get_duplicate_box_mask(filtered_objects)
        detection_results_to_return = []
        for object_index in range(len(filtered_objects)):
            if filtered_mask[object_index] == True:
                # get all values from the filtered object list
                xmin = filtered_objects[object_index][0]
                ymin = filtered_objects[object_index][1]
                xmax = filtered_objects[object_index][2]
                ymax = filtered_objects[object_index][3]
                confidence = filtered_objects[object_index][4]
                class_id = filtered_objects[object_index][5]
                detection_results_to_return.append(filtered_objects[object_index])
    
        return detection_results_to_return
    
    
    def __parseTinyYoloV3Output(self, output_node_results,  # type: numpy.ndarray
                                      filtered_objects,     # type: List
                                      original_image):      # type: OpenCV Mat
        """ Parse tiny yolo v3 inference results by filtering out duplicates via iou
            comparisons, and then filtering out low scoring results. """
            
        source_image_width = float(original_image.shape[1])
        source_image_height = float(original_image.shape[0])
        scaled_w = int(source_image_width * min(self.input_width/source_image_width, self.input_width/source_image_height))
        scaled_h = int(source_image_height * min(self.input_height/source_image_width, self.input_height/source_image_height))

        # transpose the output node results
        output_node_results = output_node_results.transpose(0,2,3,1)
        output_h = output_node_results.shape[1]
        output_w = output_node_results.shape[2]

        # 80 class scores + 4 coordinate values + 1 objectness score = 85 values
        # 85 values * 3 prior box scores per grid cell= 255 values 
        # 255 values * either 26 or 13 grid cells
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
                box_confidence_score = output_node_results[0][row][col][anchor_boxes * self.num_of_classes + 5 + 4]
                if box_confidence_score < self.detection_threshold:
                    continue
                # Calculate the x, y, width, and height of the box
                x_center = (col + output_node_results[0][row][col][anchor_boxes * self.num_of_classes + 5 + 0]) / output_w * scaled_w
                y_center = (row + output_node_results[0][row][col][anchor_boxes * self.num_of_classes + 5 + 1]) / output_h * scaled_h
                width = numpy.exp(output_node_results[0][row][col][anchor_boxes * self.num_of_classes + 5 + 2]) * self.anchors[anchor_offset + 2 * anchor_boxes]
                height = numpy.exp(output_node_results[0][row][col][anchor_boxes * self.num_of_classes + 5 + 3]) * self.anchors[anchor_offset + 2 * anchor_boxes + 1]
                # Now we check for anchor box for the highest class probabilities.
                # If the probability exceeds the threshold, we save the box coordinates, class score and class id
                for class_id in range(self.num_of_classes): 
                    class_probability = output_node_results[0][row][col][anchor_boxes * self.num_of_classes + 5 + 5 + class_id]
                    # Calculate the class's confidence score by multiplying the box_confidence score by the class probabiity
                    class_confidence_score = class_probability * box_confidence_score
                    if class_confidence_score < self.detection_threshold:
                        continue
                    # Calculate the bounding box top left and bottom right vertexes
                    xmin = max(int((x_center - width / 2) * x_ratio), 0)
                    ymin = max(int((y_center - height / 2) * y_ratio), 0)
                    xmax = min(int(xmin + width * x_ratio), source_image_width-1)
                    ymax = min(int(ymin + height * y_ratio), source_image_height-1)
                    filtered_objects.append((xmin, ymin, xmax, ymax, class_confidence_score, class_id))
        

    
    def __get_duplicate_box_mask(self, box_list # type: List
                                ):
        # type: (...) -> List
        """
        Creates a mask to remove duplicate objects (boxes) and their related probabilities and classifications
        that should be considered the same object.  This is determined by how similar the boxes are
        based on the intersection-over-union metric.
        
        box_list is as list of boxes (4 floats for centerX, centerY and Length and Width)
        
        The intersection-over-union threshold to use when determining duplicates.
        objects/boxes found that are over this threshold will be
        considered the same object.
        Returns a List of boolean True if objects were found/filtered. 
        """
        max_iou = self.iou_threshold

        box_mask = numpy.ones(len(box_list))

        # Calculate the intersection-over-union for the two boxes and set the indexes of the box_mask to 0 
        # if they exceed the max_iou threshold
        for i in range(len(box_list)):
            if box_mask[i] == 0: continue
            for j in range(i + 1, len(box_list)):
                if self.__get_intersection_over_union(box_list[i], box_list[j]) >= max_iou:
                    if box_list[i][4] < box_list[j][4]:
                        box_list[i], box_list[j] = box_list[j], box_list[i]
                    box_mask[j] = 0.0

        # convert the box_mask binary values to bool
        filter_iou_mask = numpy.array(box_mask > 0.0, dtype='bool')
        
        return filter_iou_mask



    def __get_intersection_over_union(self, box_1, # type: numpy.ndarray
                                            box_2  # type: numpy.ndarray
                                     ):
        # type: (...) -> float
        """
        Evaluate the intersection-over-union for two boxes.
        The intersection-over-union metric determines how close
        two boxes are to being the same box.  The closer the boxes
        are to being the same, the closer the metric will be to 1.0
        
        box_1 and box_2 are arrays of 4 numbers which are the (x, y)
        points that define the center of the box and the length and width of
        the box.
        Returns the intersection-over-union (between 0.0 and 1.0)
        for the two boxes specified.
        """
        print type(box_1)
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
        
        
    def set_parameter(self, tag,   # type: str
                            value  # type: Any
                     ):
        """ Sets various network parameters.
            Available parameters: detection_threshold, iou_threshold, num_of_classes """
        
        if tag == "detection_threshold":
            self.detection_threshold = value
        elif tag == "iou_threshold":
            self.iou_threshold = value
        elif tag == "num_of_classes":
            self.num_of_classes = value
            
