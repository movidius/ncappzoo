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
TEST_IMAGES_DIR = './test_faces/'

VALIDATED_IMAGES_DIR = './validated_face/'
validated_image_filename = VALIDATED_IMAGES_DIR + 'valid_face.png'

ir = "20180408-102900.xml"
DEVICE = "MYRIAD"

# name of the opencv window
CV_WINDOW_NAME = "FaceNet"

CAMERA_INDEX = 0
REQUEST_CAMERA_WIDTH = 640
REQUEST_CAMERA_HEIGHT = 480

# the same face will return 0.0
# different faces return higher numbers
# this is NOT between 0.0 and 1.0
FACE_MATCH_THRESHOLD = 0.91
network_input_h = 0
network_input_w = 0

# Run an inference on the passed image
# - image_to_classify is the image on which an inference will be performed
#    upon successful return this image will be overlayed with boxes
#    and labels identifying the found objects within the image.
# - facenet_exec_net is the executable network object that will
#    be used to peform the inference.
# - input and output blob are the input and output node names
def run_inference(image_to_classify, facenet_exec_net, input_blob, output_blob):

    # ***************************************************************
    # Send the image to the NCS
    # ***************************************************************
    results = facenet_exec_net.infer({input_blob: image_to_classify})


    return results[output_blob].flatten()


# overlays the boxes and labels onto the display image.
# - display_image is the image on which to overlay to
# - image info is a text string to overlay onto the image.
# - matching is a Boolean specifying if the image was a match.
# returns None
def overlay_on_image(display_image, image_info, matching):
    rect_width = 10
    offset = int(rect_width/2)
    if (image_info != None):
        cv2.putText(display_image, image_info, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
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
def preprocess_image(src):
    # scale the image
    global network_input_w, network_input_h
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
def run_images(valid_output, validated_image_filename, facenet_exec_net, input_image_filename_list, input_blob, output_blob):
    print("------------------ " + YELLOW + "Facenet" + NOCOLOR + " ------------------\n")
    print(" - Face Match Threshold: " + YELLOW + str(FACE_MATCH_THRESHOLD) + NOCOLOR)
    print(" - Valid image: " + YELLOW + validated_image_filename + NOCOLOR)
    print(" - Test images: " + YELLOW + TEST_IMAGES_DIR + str(input_image_filename_list) + NOCOLOR)
    print("\n---------------------------------------------\n")
    cv2.namedWindow(CV_WINDOW_NAME)
    for input_image_file in input_image_filename_list :
        # read one of the images to run an inference on from the disk
        infer_image = cv2.imread(TEST_IMAGES_DIR + input_image_file)
        if infer_image is None:
            print("Cannot read image.")
            exit(1)
        # run a single inference on the image and overwrite the
        # boxes and labels
        
        preprocessed_image = preprocess_image(infer_image)
        test_output = run_inference(preprocessed_image, facenet_exec_net, input_blob, output_blob)
        # scale the faces so that we can display a large enough image in the window
        infer_image_h = infer_image.shape[0]
        infer_image_w = infer_image.shape[1]
        # h to w ratio of original image
        h_w_ratio = infer_image_h / infer_image_w
        # calculate new h and w
        new_infer_image_w = 300
        new_infer_image_h = int(new_infer_image_w * h_w_ratio)
        # resize for better viewing
        infer_image = cv2.resize(infer_image, (new_infer_image_w, new_infer_image_h))
        
        # Test the inference results of this image with the results
        # from the known valid face.
        matching = False
        if (face_match(valid_output, test_output)):
            matching = True
            text_color = (0, 255, 0)
            match_text = "MATCH"
            print(GREEN + ' PASS!  File ' + input_image_file + ' matches ' + validated_image_filename + "\n" + NOCOLOR)
        else:
            matching = False
            match_text = "NOT A MATCH"
            text_color = (0, 0, 255)
            print(RED + ' FAIL!  File ' + input_image_file + ' does not match ' + validated_image_filename + "\n" + NOCOLOR)

        overlay_on_image(infer_image, input_image_file, matching)
	
        cv2.putText(infer_image, match_text + " - Hit key for next.", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # check if the window is visible, this means the user hasn't closed
        # the window via the X button
        prop_val = cv2.getWindowProperty(CV_WINDOW_NAME, cv2.WND_PROP_ASPECT_RATIO)
        if (prop_val < 0.0):
            print('window closed')
            break

        # display the results and wait for user to hit a key
        cv2.imshow(CV_WINDOW_NAME, infer_image)
        cv2.waitKey(0)


# This function is called from the entry point to do
# all the work of the program
def main():
    global network_input_h, network_input_w
    ie = IECore()
    net = IENetwork(model = ir, weights = ir[:-3] + 'bin')
    input_blob = next(iter(net.inputs))
    output_blob = next(iter(net.outputs))
    
    exec_net = ie.load_network(network = net, device_name = DEVICE)
    n, c, network_input_h, network_input_w = net.inputs[input_blob].shape

    # Read the image
    validated_image = cv2.imread(validated_image_filename)
    if validated_image is None:
    	print("Cannot read image.")
    	exit(1)
    	
    # Preprocess the image
    preprocessed_image = preprocess_image(validated_image)
    
    # Run the inference
    valid_output = run_inference(preprocessed_image, exec_net, input_blob, output_blob)

    # Get list of all the .jpg files in the image directory
    input_image_filename_list = os.listdir(TEST_IMAGES_DIR)
    input_image_filename_list = [i for i in input_image_filename_list if i.endswith('.png')]
    if (len(input_image_filename_list) < 1):
        # no images to show
        print('No .png files found')
        return 1
    else:
        print("images: " + str(input_image_filename_list) + '\n')
    
    # Run the inferences and make comparisons 
    run_images(valid_output, validated_image_filename, exec_net, input_image_filename_list, input_blob, output_blob)
    print(" Finished.\n")



# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())
