#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.


from mvnc import mvncapi as mvnc
import numpy
import cv2
import sys
import os

EXAMPLES_BASE_DIR='../../'
#IMAGES_DIR = EXAMPLES_BASE_DIR + 'data/images/'
#IMAGE_FULL_PATH = IMAGES_DIR + 'nps_chair.png'
#IMAGE_FULL_PATH = './neal_pic.jpg'
#IMAGE_FULL_PATH = './Priscilla-Presley-Biography.jpg'
IMAGES_DIR = './'

VALIDATED_IMAGES_DIR = IMAGES_DIR + 'validated_images/'
validated_image_filename = VALIDATED_IMAGES_DIR + 'valid.jpg'

GRAPH_FILENAME = "facenet_celeb.graph"

# name of the opencv window
CV_WINDOW_NAME = "Facenet - hit any key to exit"

CAMERA_INDEX = 1
REQUEST_CAMERA_WIDTH = 640
REQUEST_CAMERA_HEIGHT = 480

# the same face will return 0.0
# different faces return higher numbers
FACE_MATCH_THRESHOLD = 1.0

# ***************************************************************
# Labels for the classifications for the network.
# ***************************************************************
LABELS = ('background',
          'aeroplane', 'bicycle', 'bird', 'boat',
          'bottle', 'bus', 'car', 'cat', 'chair',
          'cow', 'diningtable', 'dog', 'horse',
          'motorbike', 'person', 'pottedplant',
          'sheep', 'sofa', 'train', 'tvmonitor')


# Run an inference on the passed image
# image_to_classify is the image on which an inference will be performed
#    upon successful return this image will be overlayed with boxes
#    and labels identifying the found objects within the image.
# ssd_mobilenet_graph is the Graph object from the NCAPI which will
#    be used to peform the inference.
def run_inference(image_to_classify, facenet_graph, image_filename):

    # get a resized version of the image that is the dimensions
    # SSD Mobile net expects
    resized_image = preprocess_image(image_to_classify)

    #cv2.imshow("preprocessed", resized_image)

    # ***************************************************************
    # Send the image to the NCS
    # ***************************************************************
    facenet_graph.LoadTensor(resized_image.astype(numpy.float16), None)

    # ***************************************************************
    # Get the result from the NCS
    # ***************************************************************
    output, userobj = facenet_graph.GetResult()

    #print("Total results: " + str(len(output)))
    #print(output)

    overlay_on_image(image_to_classify, output, image_filename)

    return output


# overlays the boxes and labels onto the display image.
# display_image is the image on which to overlay the boxes/labels
# object_info is a list of 7 values as returned from the network
#     These 7 values describe the object found and they are:
#         0: image_id (always 0 for myriad)
#         1: class_id (this is an index into labels)
#         2: score (this is the probability for the class)
#         3: box left location within image as number between 0.0 and 1.0
#         4: box top location within image as number between 0.0 and 1.0
#         5: box right location within image as number between 0.0 and 1.0
#         6: box bottom location within image as number between 0.0 and 1.0
# returns None
def overlay_on_image(display_image, object_info, image_filename):
    cv2.putText(display_image, image_filename, (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)

def to_rgb(img):
    w, h = img.shape
    ret = numpy.empty((w, h, 3), dtype=numpy.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

# whiten an image
def whiten_image(source_image):
    source_mean = numpy.mean(source_image)
    source_standard_deviation = numpy.std(source_image)
    std_adjusted = numpy.maximum(source_standard_deviation, 1.0 / numpy.sqrt(source_image.size))
    whitened_image = numpy.multiply(numpy.subtract(source_image, source_mean), 1 / std_adjusted)
    return whitened_image

# create a preprocessed image from the source image that matches the
# network expectations and return it
def preprocess_image(src):
    # scale the image
    NETWORK_WIDTH = 160
    NETWORK_HEIGHT = 160
    preprocessed_image = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))

    #convert to RGB
    preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)

    #whiten
    preprocessed_image = whiten_image(preprocessed_image)

    # return the preprocessed image
    return preprocessed_image

# determine if two images are of matching faces based on the
# the network output for both images.
def face_match(face1_output, face2_output):
    if (len(face1_output) != len(face2_output)):
        print('length mismatch in face_match')
        return False
    total_diff = 0
    for output_index in range(0, len(face1_output)):
        this_diff = numpy.square(face1_output[output_index] - face2_output[output_index])
        total_diff += this_diff
    print('Total Difference is: ' + str(total_diff))

    if (total_diff < FACE_MATCH_THRESHOLD):
        return True

    return False

# handles key presses
# raw_key is the return value from cv2.waitkey
# returns False if program should end, or True if should continue
def handle_keys(raw_key):
    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
        return False

    return True

def run_camera(valid_output, validated_image_filename, graph):
    camera_device = cv2.VideoCapture(CAMERA_INDEX)
    camera_device.set(cv2.CAP_PROP_FRAME_WIDTH, REQUEST_CAMERA_WIDTH)
    camera_device.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_CAMERA_HEIGHT)

    actual_camera_width = camera_device.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_camera_height = camera_device.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print ('actual camera resolution: ' + str(actual_camera_width) + ' x ' + str(actual_camera_height))

    if ((camera_device == None) or (not camera_device.isOpened())):
        print ('Could not open camera.  Make sure it is plugged in.')
        print ('Also, if you installed python opencv via pip or pip3 you')
        print ('need to uninstall it and install from source with -D WITH_V4L=ON')
        print ('Use the provided script: install-opencv-from_source.sh')
        return

    frame_count = 0

    cv2.namedWindow(CV_WINDOW_NAME)

    found_match = False

    while True :
        # Read image from camera,
        ret_val, vid_image = camera_device.read()
        if (not ret_val):
            print("No image from camera, exiting")
            break

        frame_count += 1
        frame_name = 'camera frame ' + str(frame_count)

        # run a single inference on the image and overwrite the
        # boxes and labels
        test_output = run_inference(vid_image, graph, frame_name)

        if (face_match(valid_output, test_output)):
            print('PASS!  File ' + frame_name + ' matches ' + validated_image_filename)
            found_match = True
            break
        else:
            print('FAIL!  File ' + frame_name + ' does not match ' + validated_image_filename)

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

    if (found_match):
        cv2.imshow(CV_WINDOW_NAME, vid_image)
        cv2.waitKey(0)


def run_images(valid_output, validated_image_filename, graph, input_image_filename_list):
    for input_image_file in input_image_filename_list :
        # read the image to run an inference on from the disk
        infer_image = cv2.imread(input_image_file)

        # run a single inference on the image and overwrite the
        # boxes and labels
        test_output = run_inference(infer_image, graph, input_image_file)

        if (face_match(valid_output, test_output)):
            print('PASS!  File ' + input_image_file + ' matches ' + validated_image_filename)
        else:
            print('FAIL!  File ' + input_image_file + ' does not match ' + validated_image_filename)

        # display the results and wait for user to hit a key
        cv2.imshow(CV_WINDOW_NAME, infer_image)
        cv2.waitKey(0)


# This function is called from the entry point to do
# all the work of the program
def main():

    # Get a list of ALL the sticks that are plugged in
    # we need at least one
    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
        print('No NCS devices found')
        quit()

    # Pick the first stick to run the network
    device = mvnc.Device(devices[0])

    # Open the NCS
    device.OpenDevice()

    # The graph file that was created with the ncsdk compiler
    graph_file_name = GRAPH_FILENAME

    # read in the graph file to memory buffer
    with open(graph_file_name, mode='rb') as f:
        graph_in_memory = f.read()

    # create the NCAPI graph instance from the memory buffer containing the graph file.
    graph = device.AllocateGraph(graph_in_memory)

    validated_image = cv2.imread(validated_image_filename)
    valid_output = run_inference(validated_image, graph, validated_image_filename)

    #run with camera
    #run_camera(valid_output, validated_image_filename, graph)

    # get list of all the .jpg files in the image directory
    input_image_filename_list = os.listdir(IMAGES_DIR)
    input_image_filename_list = [i for i in input_image_filename_list if i.endswith('.jpg')]
    if (len(input_image_filename_list) < 1):
        # no images to show
        print('No .jpg files found')
        return 1
    run_images(valid_output, validated_image_filename, graph, input_image_filename_list)


    # Clean up the graph and the device
    graph.DeallocateGraph()
    device.CloseDevice()


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())