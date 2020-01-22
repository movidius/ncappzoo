from openvino.inference_engine import IENetwork, IECore
import numpy
import cv2
import sys
sys.path.append('../../shared/Python/')
from Object_detector import *

class Ssd_mobilenet_object_detector(Object_detector):
    def __init__(self, ie,              # type: IECore 
                       net,             # type: IENetwork
                       device="MYRIAD"  # type: str
                ):
    	""" Constructor for Ssd_mobilenet_object_detector."""
        # Get the input and output node names
        self.input_blob = next(iter(net.inputs))
        self.output_blob = next(iter(net.outputs))

        # Get the input and output shapes from the input/output nodes
        self.input_shape = net.inputs[self.input_blob].shape
        self.output_shape = net.outputs[self.output_blob].shape
        self.input_batchsize, self.input_channels, self.input_height, self.input_width = self.input_shape
        self.output_x, self.output_y, self.detections_count, self.detections_size = self.output_shape
        self.exec_net = ie.load_network(network = net, device_name = device)
        self.detection_threshold=0.70


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
        # send the image for preprocessing 
        preprocessed_image = self.__preprocess_image(original_image)
        cur_request_id = 0

        request_handle = self.exec_net.start_async(request_id=cur_request_id, inputs={self.input_blob: preprocessed_image})

        # wait for inference to complete
        if self.exec_net.requests[cur_request_id].wait(-1) == 0:
            results_to_return = self.__postprocess(request_handle, original_image)
            return results_to_return


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
        source_image_width = float(original_image.shape[1])
        source_image_height = float(original_image.shape[0])
        # Extract the inference data for the output node name
        inference_results = request_handle.outputs[self.output_blob]
        # Process the results
        detection_results_to_return = []
        for num, detection_result in enumerate(inference_results[0][0]):
            # Save only detection_results when probability > than specified threshold
            if detection_result[2] > self.detection_threshold:
                box_left = int(detection_result[3] * source_image_width)
                box_top = int(detection_result[4] * source_image_height)
                box_right = int(detection_result[5] * source_image_width)
                box_bottom = int(detection_result[6] * source_image_height)
                class_confidence_score = float(detection_result[2])
                class_id = int(detection_result[1])
                detection_results_to_return.append((box_left, box_top, box_right, box_bottom, class_confidence_score, class_id))
        return detection_results_to_return
        
        
    def set_parameter(self, tag,   # type: str 
                            value  # type: Any
                     ):
        """ Sets model parameters. 
            Available parameters: detection_threshold """
        if tag == "detection_threshold":
            self.detection_threshold = value
        
        
        

