from openvino.inference_engine import IENetwork, IECore

class Object_detector:    
    def __init__(self, ie,      # type: IECore 
                       net,     # type: IENetwork
                       device   # type: str
                ):        
        """ Constructor for object detector processor. 
            Does 3 main things:           
            1. Get input and output layer names for the model. 
            2. Gather input and output shapes (n,c,h,w) from the model input/output and input/output blobs.
            3. Create the ExecutableNetwork object by specifying the IENetwork object and device name (in this case MYRIAD). """
        pass
    
    
    def __preprocess_image(self, src_image # type: OpenCV Mat
                          ):
        # type: (...) -> OpenCV Mat
        """ Perform image preprocessing. Typically it is an image resize, 
            then a transpose (HWC -> CHW) if using OpenCV 
            to read the image. Returns preprocessed image (OpenCV Mat if using OpenCV). """
        pass
        
    
    def run_inference_sync(self, original_image # type: OpenCV Mat
                          ):
        # type: (...) -> List
        """ Preprocess image, runs inference, does postprocessing on inference results. Returns results as a List.
            1. Calls __preprocess_image() to preprocess the image. 
            2. Run an inference on the NCS using the preprocessed image as input. 
            3. Call __postprocess_results to organize the data. 
            4. Return the results as a list of tuples (4 bounding box coordinates, 
               confidence score, and class id). """
        pass
        
        
    def __postprocess_results(self, results_handle # type: InferRequest
                             ):
        # type: (...) -> List
        """ Organize the network inference results and return the results.
            The results will be returned as a list of tuples in the following format: 
            (Left side of bbox, Top side of bbox, Right side of bbox, Bottom of the bbox, Confidence score, Class id)
            Pseudocode Example:
                all_results = do_inference()
                results_to_return = []
                
                for result in all_results:
                    results_to_return.append((result.left_box, result.top_box, result.right_box, result.bottom_box, result.confidence_score, result.class_id))
                
                return results_to_return """
        pass
        
        
    def set_parameters(self, tag,  # type: str
                             value # type: Any
                      ):
        """ Sets a network parameter. """
        pass
        
