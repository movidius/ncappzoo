#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.

from openvino.inference_engine import IENetwork, IECore
import sys
import numpy
import cv2
import os

GREEN = '\033[1;32m'
RED = '\033[1;31m'
NOCOLOR = '\033[0m'
YELLOW = '\033[1;33m'

EXAMPLES_BASE_DIR='../../'

VALIDATED_IMAGES_DIR = './validated_faces/'

facenet_xml = "20180408-102900.xml"
fd_xml = "face-detection-retail-0004.xml"

DEVICE = "MYRIAD"

# name of the opencv window
CV_WINDOW_NAME = "video_face_matcher"

CAMERA_INDEX = 0
REQUEST_CAMERA_WIDTH = 640
REQUEST_CAMERA_HEIGHT = 480

# the same face will return 0.0
# different faces return higher numbers
# this is NOT between 0.0 and 1.0
FACE_MATCH_THRESHOLD = 0.91
FACE_DETECTION_THRESHOLD = 0.75
network_input_h = 0
network_input_w = 0

# Run an inference on the passed image
# - image_to_classify is the image on which an inference will be performed
#    upon successful return this image will be overlayed with boxes
#    and labels identifying the found objects within the image.
# - facenet_exec_net is the executable network object that will
#    be used to peform the inference.
# - input and output blob are the input and output node names
def run_inference(image_to_classify, exec_net, input_blob, output_blob):

    # ***************************************************************
    # Send the image to the NCS
    # ***************************************************************
    results = exec_net.infer({input_blob: image_to_classify})


    return results[output_blob].flatten()


# overlays the boxes and labels onto the display image.
# - display_image is the image on which to overlay to
# - image info is a text string to overlay onto the image.
# - matching is a Boolean specifying if the image was a match.
# returns None
def overlay_on_image(display_image, matching):
    rect_width = 10
    offset = int(rect_width/2)
    if (matching):
        # match, green rectangle
        cv2.rectangle(display_image, (0+offset, 0+offset),
                      (display_image.shape[1]-offset-1, display_image.shape[0]-offset-1),
                      (0, 255, 0), 10)
    else:
        # not a match, red rectangle
        cv2.rectangle(display_image, (0+offset, 0+offset),
                      (display_image.shape[1]-offset-1, display_image.shape[0]-offset-1),
                      (0, 0, 255), 10)


# Whiten an image
def whiten_image(source_image):
    source_mean = numpy.mean(source_image)
    source_standard_deviation = numpy.std(source_image)
    std_adjusted = numpy.maximum(source_standard_deviation, 1.0 / numpy.sqrt(source_image.size))
    whitened_image = numpy.multiply(numpy.subtract(source_image, source_mean), 1 / std_adjusted)
    return whitened_image


# Create a preprocessed image from the source image that matches the
# network expectations and return it
def preprocess_image(src, network_input_w, network_input_h):
    # scale the image
    preprocessed_image = cv2.resize(src, (network_input_w, network_input_h))
    preprocessed_image = numpy.transpose(preprocessed_image)
    preprocessed_image = numpy.reshape(preprocessed_image, (1, 3, network_input_w, network_input_h))
    #whiten (not used)
    #preprocessed_image = whiten_image(preprocessed_image)

    # return the preprocessed image
    return preprocessed_image


# determine if two images are of matching faces based on the
# the network output for both images.
def face_match(face1_output, face2_output):
    if (len(face1_output) != len(face2_output)):
        print('length mismatch in face_match')
        return False
    total_diff = 0
    # Sum of all the squared differences
    for output_index in range(0, len(face1_output)):
        this_diff = numpy.square(face1_output[output_index] - face2_output[output_index])
        total_diff += this_diff
	# Now take the sqrt to get the L2 difference
    total_diff = numpy.sqrt(total_diff)
    print(' Total Difference is: ' + str(total_diff))
    if (total_diff < FACE_MATCH_THRESHOLD):
        # the total difference between the two is under the threshold so
        # the faces match.
        return True

    # differences between faces was over the threshold above so
    # they didn't match.
    return False


# Handles key presses
# - raw_key is the return value from cv2.waitkey
# returns False if program should end, or True if should continue
def handle_keys(raw_key):
    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
        return False

    return True


# Test all files in a list for a match against a valided face and display each one.
# valid_output is inference result for the valid image
# validated image filename is the name of the valid image file
# facenet_exec_net is the executable network object
#   which we will run the inference on.
# input_image_filename_list is a list of image files to compare against the
#   valid face output.
def run_camera(valid_output, validated_image_filename, facenet_exec_net, facenet_input_blob, facenet_output_blob):
    print("------------------ " + YELLOW + "Facenet" + NOCOLOR + " ------------------\n")
    print(" - Face Match Threshold: " + YELLOW + str(FACE_MATCH_THRESHOLD) + NOCOLOR)
    print(" - Valid image: " + YELLOW + validated_image_filename + NOCOLOR)
    print("\n---------------------------------------------\n")
    camera_device = cv2.VideoCapture(CAMERA_INDEX)
    camera_device.set(cv2.CAP_PROP_FRAME_WIDTH, REQUEST_CAMERA_WIDTH)
    camera_device.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_CAMERA_HEIGHT)

    actual_camera_width = camera_device.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_camera_height = camera_device.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cv2.namedWindow(CV_WINDOW_NAME)
    
    while (True):
    
        # Read image from camera,
        ret_val, vid_image = camera_device.read()
        if (not ret_val):
            print("No image from camera, exiting")
            break
        
        preprocessed_image = preprocess_image(vid_image)
        test_output = run_inference(preprocessed_image, facenet_exec_net, facenet_input_blob, facenet_output_blob)
        
        # Test the inference results of this image with the results
        # from the known valid face.
        matching = False
        if (face_match(valid_output, test_output)):
            matching = True
            text_color = (0, 255, 0)
            match_text = "MATCH"
            #print(GREEN + ' PASS!  File ' + input_image_file + ' matches ' + validated_image_filename + "\n" + NOCOLOR)
        else:
            matching = False
            match_text = "NOT A MATCH"
            text_color = (0, 0, 255)
            #print(RED + ' FAIL!  File ' + input_image_file + ' does not match ' + validated_image_filename + "\n" + NOCOLOR)

        overlay_on_image(vid_image, matching)
	    
        cv2.putText(vid_image, match_text + " - Hit key for next.", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # check if the window is visible, this means the user hasn't closed
        # the window via the X button
        prop_val = cv2.getWindowProperty(CV_WINDOW_NAME, cv2.WND_PROP_ASPECT_RATIO)
        if (prop_val < 0.0):
            print('window closed')
            break

        # display the results and wait for user to hit a key
        cv2.imshow(CV_WINDOW_NAME, vid_image)
        raw_key = cv2.waitKey(1)
        if (raw_key != -1):
            if (handle_keys(raw_key) == False):
                print('user pressed Q')
                break



def fd_post_processing(fd_results):
    detected_faces = []
    for face_num, detection_result in enumerate(fd_results):
        # Draw only detection_resultects when probability more than specified threshold
        if detection_result[2] > detection_threshold:
            box_left = int(detection_result[3] * image_w)
            box_top = int(detection_result[4] * image_h)
            box_right = int(detection_result[5] * image_w)
            box_bottom = int(detection_result[6] * image_h)
            class_id = int(detection_result[1])
            
            cropped_face = frame[box_top:box_bottom, box_left:box_right]
            detected_faces.append((cropped_face, box_left, box_top, box_right, box_bottom))

    return detected_faces


def read_all_validated_faces():
    #input_image_filename_list = os.listdir(VALIDATED_IMAGES_DIR)
    people_and_faces = []
    for root, sub_dirs, files in os.walk(VALIDATED_IMAGES_DIR):
        #print("root:", root)
        #print("sub_dirs:", sub_dirs)
        #print("files:", files)
        if len(files) > 0:
            for images in files:
                person_name = root[len(VALIDATED_IMAGES_DIR):]
                file_name = str(root) + "/" + str(images)
                image_mat = cv2.imread(file_name)
                people.append((person_name, file_name, image_mat))

    return people_and_faces

# This function is called from the entry point to do
# all the work of the program
def main():
    global network_input_h, network_input_w
    ie = IECore()
    # facenet network prep
    facenet = IENetwork(model = facenet_xml, weights = facenet_xml[:-3] + 'bin')
    facenet_input_blob = next(iter(facenet.inputs))
    facenet_output_blob = next(iter(facenet.outputs))
    # facenet load network and get the network shape information
    facenet_exec_net = ie.load_network(network = facenet, device_name = DEVICE)
    facenet_batch_size, facenet_channels, facenet_input_h, facenet_input_w = facenet.inputs[facenet_input_blob].shape

    # face detection network prep
    fd_net = IENetwork(model = fd_xml, weights = fd_xml[:-3] + 'bin')
    fd_input_blob = next(iter(fd_net.inputs))
    fd_output_blob = next(iter(fd_net.outputs))

    # Load the network and get the network shape information
    fd_exec_net = ie.load_network(network = fd_net, device_name = DEVICE)
    fd_n, fd_c, fd_h, fd_w = fd_net.inputs[fd_input_blob].shape
    fd_x, fd_y, fd_count, fd_size = fd_net.outputs[fd_output_blob].shape


    people_faces = read_all_validated_faces()
    print("number of faces: ", len(people_faces))
    for face in people_faces:
        fd_preprocessed_image = preprocess_image(face[2], facenet_input_w, facenet_input_h)
        people_faces_preprocessed.append((faces[0], faces[1], preprocessed_face))
    exit(0)
    # Read the validated images
    validated_image = cv2.imread(validated_image_filename)
    if validated_image is None:
    	print("Cannot read image.")
    	exit(1)
    	
    # Preprocess the image for face detection
    fd_preprocessed_image = preprocess_image(validated_image, facenet_input_w, facenet_input_h)
    fd_results = run_inference(fd_preprocessed_image, fd_exec_net, fd_input_blob, fd_output_blob)
    detected_faces = fd_post_processing(fd_results)
    
    
    # Preprocess the image for facenet
    facenet_preprocessed_image = preprocess_image(validated_image, facenet_input_w, facenet_input_h)
    
    
    # Run the inference and get the validated face embeddings
    valid_output = run_inference(preprocessed_image, facenet_exec_net, facenet_input_blob, facenet_output_blob)

    # Run the inferences and make comparisons
    run_camera(valid_output, validated_image_filename, facenet_exec_net, facenet_input_blob, facenet_output_blob)
    print(" Finished.\n")


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())

