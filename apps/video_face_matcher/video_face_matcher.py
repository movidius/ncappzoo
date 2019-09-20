#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.

from openvino.inference_engine import IENetwork, IECore
import sys
import numpy
import cv2
import os
from tkinter import *

from Face_detection_results import *
from Face_profile import *
from Person_profile import *
from My_network import *

GREEN = '\033[1;32m'
RED = '\033[1;31m'
NOCOLOR = '\033[0m'
YELLOW = '\033[1;33m'

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
FACE_MATCH_THRESHOLD = 1.10
FACE_DETECTION_THRESHOLD = 0.50

FONT_SCALE = 1
FONT_THICKNESS = 2
mouse_down = False
entry = None


def run_inference(image_to_classify, exec_net, input_node_name, output_node_name):
    '''
    Runs an inference on an image.
    
    Returns an inference result (differs for every network).
    '''
    results = exec_net.infer({input_node_name: image_to_classify})
    return results[output_node_name]


def overlay_on_image(display_image, matching, box_left, box_top, box_right, box_bottom, match_text):
    ''' 
    Overlays a bounding box on the display image. Green if a match, Red if not a match. 
    
    Returns None
    '''
    rect_width = 2
    label_background_color = (70, 120, 70)
    # Draws a rectangle with a greenish background for the text
    label_size = cv2.getTextSize(match_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)[0]
    label_left = box_left
    label_top = box_top - label_size[1]
    label_right = label_left + label_size[0]
    label_bottom = label_top + label_size[1]
    cv2.rectangle(display_image,(label_left-1, label_top-7),(label_right+1, label_bottom+1), label_background_color, cv2.FILLED)
    
    if (matching):
        # match, green rectangle
        cv2.rectangle(display_image, (box_left, box_top),
                      (box_right, box_bottom),
                      (0, 255, 0), rect_width)
    else:
        # not a match, red rectangle
        cv2.rectangle(display_image, (box_left, box_top),
                      (box_right, box_bottom),
                      (0, 0, 255), rect_width)


def preprocess_image(src, network_input_w, network_input_h):
    '''
    Preprocesses the image by resizing the image, doing a transpose and reshaping the tensor.
    
    Returns the preprocessed image.
    '''
    # Resize and transposes the image
    preprocessed_image = cv2.resize(src, (network_input_w, network_input_h))
    preprocessed_image = numpy.transpose(preprocessed_image, (2,0,1))
    preprocessed_image = numpy.reshape(preprocessed_image, (1, 3, network_input_w, network_input_h))
    
    return preprocessed_image


def face_match(face1_output, face2_output):
    ''' 
    Compares two feature vectors by taking the sum of the squared differences.
    
    Returns the total difference between the vectors or 99 if the length of the faces don't match.
    '''
    if (len(face1_output) != len(face2_output)):
        print('length mismatch in face_match')
        return 99
    total_diff = 0
    # Sum of all the squared differences
    for output_index in range(0, len(face1_output)):
        this_diff = numpy.square(face1_output[output_index] - face2_output[output_index])
        total_diff += this_diff
	# Now take the sqrt to get the L2 difference
    total_diff = numpy.sqrt(total_diff)
    #print(' Total Difference is: ' + str(total_diff))
    
    return total_diff


def check_key_exit_event(raw_key):
    ''' Checks if user pressed the exit key. '''
    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
        return False
    return True


def fd_post_processing(fd_results, frame, image_h, image_w):
    ''' 
    Performs face detection retail 0004 network post processing
    
    Returns a list of Face_detection_result objects that include the cropped face OpenCV Mat and bounding box coordinates.
    '''
    global FACE_DETECTION_THRESHOLD
    detection_threshold = FACE_DETECTION_THRESHOLD
    detected_faces = []
    
    for face_num, detection_result in enumerate(fd_results[0][0]):
        # Draw only detection_resultects when probability more than specified threshold
        if detection_result[2] > detection_threshold:
            box_left = max(int(detection_result[3] * image_w), 0)
            box_top = max(int(detection_result[4] * image_h), 0)
            box_right = min(int(detection_result[5] * image_w), image_w)
            box_bottom = min(int(detection_result[6] * image_h), image_h)
            class_id = int(detection_result[1])
            
            cropped_face = frame[box_top:box_bottom, box_left:box_right]
            detected_faces.append(Face_detection_result(face_mat=cropped_face, box_left=box_left, box_top=box_top, box_right=box_right, box_bottom=box_bottom))

    return detected_faces


def read_all_validated_images():
    '''
    Reads in image files of people and saves the OpenCV Mat. Also saves their name.
    
    Returns a list of Face_profile objects that include the person's name, image path, and OpenCV Mat.
    '''
    names_images_mats = []
    for root, sub_dirs, files in os.walk(VALIDATED_IMAGES_DIR):
        if len(files) > 0:
            for images in files:
                person_name = root[len(VALIDATED_IMAGES_DIR):]
                file_name = str(root) + "/" + str(images)
                image_mat = cv2.imread(file_name)
                names_images_mats.append(Person_profile(name=person_name, image_path=file_name, image_mat=image_mat))

    return names_images_mats


def get_validated_faces(fd, fn, validated_person_images):
    '''
    Runs inferences on images to crop out faces using the face_detection_retail_0004 network. Then using those faces as input, run inferences using the facenet network to get a feature vector for each face. 
    
    Returns a list of Face_profile(s) that include the person's name, image path, and a feature vector.
    '''
    # Preprocess all of the face mats, then run an inference on all of the faces, then save all of the feature vectors
    valid_processed_faces = []
    for person_image in validated_person_images:
        # Preprocess a person image
        fd_preprocessed_image = preprocess_image(person_image.image_mat, fd.input_w, fd.input_h)
        # Run an inference on the person image. Will return a face with bounding box coords.
        fd_results = run_inference(fd_preprocessed_image, fd.exec_net, fd.input_node_name, fd.output_node_name)
        
        # Get the original image file size
        image_height = person_image.image_mat.shape[0]
        image_width = person_image.image_mat.shape[1]
        
        # Do post processing on the face detection results. This will return a face mat and bbox coords if faces were found.
        detected_faces = fd_post_processing(fd_results, person_image.image_mat, image_height, image_width)
        
        # We may have found a face, so lets run facenet to get the feature vector
        if (len(detected_faces) > 0):
            # Resizing the face mat for use with facenet. Note: Only 1 face per person image is used. Uses the first face.
            facenet_preprocessed_image = preprocess_image(detected_faces[0].face_mat, fn.input_w, fn.input_h)
            # Run an inference on the face to get the feature vector
            valid_face_output = run_inference(facenet_preprocessed_image, fn.exec_net, fn.input_node_name, fn.output_node_name)
            current_face_feature_vector = valid_face_output[0]
            
            # Save the name of the person, the image path and the feature vector for the face
            valid_processed_faces.append(Face_profile(name=person_image.name, image_path=person_image.image_path, feature_vector=current_face_feature_vector))
    
    return valid_processed_faces


def get_network_information(ie, xml):
    ''' 
    Reads in xml and bin files and saves all relavent network information for later use.
    Returns a My_network object that includes the ExecutableNetwork object, the network input node name, the network output node name, the network input width, and the network input height.
    '''
    # Read the network xml and bin files
    net = IENetwork(model = xml, weights = xml[:-3] + 'bin')
    # Get the network input node/layer names
    input_node_name = next(iter(net.inputs))
    output_node_name = next(iter(net.outputs))

    # Load the network and get the network shape information
    exec_net = ie.load_network(network = net, device_name = DEVICE)
    n, c, input_h, input_w = net.inputs[input_node_name].shape
    # Save the network information that we need 
    my_network = My_network(exec_net = exec_net,
                         input_node_name = input_node_name,
                         output_node_name = output_node_name,
                         input_w = input_w,
                         input_h = input_h)
    return my_network


def run_camera(valid_processed_faces, fd, fn):
    '''
    Read frames from the camera and detect faces in the camera frame by running an inference on an image using a face detection network. Afterwards run an inference on the detected faces using facenet to get a feature vector for each face. Finally, compare the feature vector of all detected faces to the feature vectors of the valid faces to find the best match if it exists.
    
    Parameters
    ----------
    valid_processed_faces (List of Face_profile) : Holds all of the "known" person face data: 
        name - name of the person, 
        image_path - relative path for the image, 
        feature_vector - a 512 embedding face feature vector
    
    fd (My_network) : Holds the network information and parameters for the face_detection_retail_0004 network.
        exec_net - OpenVINO ExecutableNetwork Object.
        input_node_name - The network's input node/layer name.
        output_node_name - The network's output node/layer name.
        input_w - The network's input shape width.
        input_h - The network's input shape height.
        
    fn (My_network) : Holds the network information and parameters for the facenet network.
        exec_net - OpenVINO ExecutableNetwork Object.
        input_node_name - The network's input node/layer name.
        output_node_name - The network's output node/layer name.
        input_w - The network's input shape width.
        input_h - The network's input shape height.
        
    Returns None
    '''
    print("------------------ " + YELLOW + "video_face_matcher" + NOCOLOR + " ------------------\n")
    print(" - Face Match Threshold: " + YELLOW + str(FACE_MATCH_THRESHOLD) + NOCOLOR)
    print("\n---------------------------------------------\n")
    
    # Camera set up
    camera_device = cv2.VideoCapture(CAMERA_INDEX)
    camera_device.set(cv2.CAP_PROP_FRAME_WIDTH, REQUEST_CAMERA_WIDTH)
    camera_device.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_CAMERA_HEIGHT)

    actual_camera_width = camera_device.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_camera_height = camera_device.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cv2.namedWindow(CV_WINDOW_NAME)
    
    
    test_processed_faces = []
    

    while (True):
        # Read an image frame from camera
        ret_val, vid_image = camera_device.read()
        if (not ret_val):
            print("No image from camera, exiting...")
            break
        vid_image = cv2.flip(vid_image, 1)
        # Preprocess the camera frame
        fd_preprocessed_image = preprocess_image(vid_image, fd.input_w, fd.input_h)
        # Run the inference to find the faces
        fd_results = run_inference(fd_preprocessed_image, fd.exec_net, fd.input_node_name, fd.output_node_name)
        
        # Get the original image's file height and width
        image_height = vid_image.shape[0]
        image_width = vid_image.shape[1]
        frame_from_vid = vid_image.copy()
        # Post processing to get the cropped face OpenCV Mat and bounding box coordinates
        detected_faces = fd_post_processing(fd_results, frame_from_vid, image_height, image_width)
        
        # Run an inference on each cropped face mat and compare it with all known validated feature vectors
        if (len(detected_faces) > 0):
            for current_face in detected_faces:
                # Preprocess the cropped face mat for use with facenet
                facenet_preprocessed_image = preprocess_image(current_face.face_mat, fn.input_w, fn.input_h)
                # Make an inference and get the feature vector for the current cropped face
                test_face_output = run_inference(facenet_preprocessed_image, fn.exec_net, fn.input_node_name, fn.output_node_name)
                
                best_score = 99 # lower is better        
                text_left = current_face.box_left # display text's left coordinate
                text_top = current_face.box_top # display text's top coordinate
                
                # Compare the current feature vector to all known validated feature vectors
                for valid_face in valid_processed_faces:
                    matching = False
                    current_face.feature_vector = test_face_output[0]
                    # Get the total difference between both feature vectors
                    total_difference = face_match( valid_face.feature_vector, current_face.feature_vector)
                    # Try to get the best match
                    if (total_difference < best_score):
                        match_text = valid_face.name
                        best_score = total_difference
                # Check to see if the best score was lower than the FACE_MATCH_THRESHOLD. lower is better.
                if (best_score <= FACE_MATCH_THRESHOLD):
                    matching = True
                    current_face.match = True
                    text_color = (0, 255, 0)
                        
                if (best_score > FACE_MATCH_THRESHOLD):
                    matching = False
                    match_text = "Unknown"
                    text_color = (0, 255, 0)
                
                # Draw the bounding box for the face
                overlay_on_image(vid_image, matching, current_face.box_left, current_face.box_top, current_face.box_right, current_face.box_bottom, match_text)
                
                # show the display text
                cv2.putText(vid_image, match_text, (text_left, text_top-5), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, text_color, FONT_THICKNESS)
        
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
            if (check_key_exit_event(raw_key) == False):
                print('user pressed Q')
                break
        # handles mouse click events
        cv2.setMouseCallback(CV_WINDOW_NAME, register_new_face, param=(detected_faces, valid_processed_faces))


def get_name(textentry, tk_window):
    '''
    Gets the user input from the tkinter window.
    Returns None
    '''
    global entry
    # Save the user's entry
    entry = textentry.get()
    # Close the window
    tk_window.destroy()
    

def register_new_face(event, x, y, flags, param):
    '''
    Handles the mouse click event to add a face to the valid faces.
    '''
    global mouse_down, entry
    detected_faces=param[0]
    validated_faces=param[1]
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_down = True
    
    # Check if mouse button was pressed
    if event == cv2.EVENT_LBUTTONUP and mouse_down:
        mouse_down = False
        for face_num, current_face in enumerate(detected_faces):
            # Checks to see mouse click was within the bounding box of any of the detected faces
            if x > current_face.box_left and x < current_face.box_right and y > current_face.box_top and y < current_face.box_bottom and current_face.match == False:
                # tkinter window set up
                tk_window = Tk()
                HEIGHT = 100
                WIDTH = 400
                # Creates the canvas for the window
                canvas = Canvas(tk_window, height=HEIGHT, width=WIDTH)
                canvas.pack()
                # Creates the frame and places it in the canvas
                frame = Frame(tk_window)
                frame.place(relwidth=1.0, relheight=1.0)
                # Creates the label and places it in the frame
                label = Label(frame, text="Enter the person's name: ")
                label.pack(side='left')
                # Creates the text entry box and places it in the frame
                textentry = Entry(frame)
                textentry.pack(side='left')
                # Creates a button and places it in the frame
                button = Button(frame, text="Submit", command=lambda: get_name(textentry, tk_window))
                button.pack(side='right')
                # Displays window
                window.mainloop()
                
                # Saves the image to disk and adds the feature vector to the valid faces
                if entry is not None:
                    os.system("mkdir -p validated_faces/" + entry)
                    max_images_per_person = 100
                    for number in range(0, max_images_per_person):
                        new_file_path = "validated_faces/" + entry + "/" + entry + str(number) + ".jpg"
                        if not (os.path.exists(new_file_path)):
                            cv2.imwrite(new_file_path, current_face.face_mat)
                            validated_faces.append(Face_profile(entry, new_file_path, current_face.feature_vector))
                            print("New image file and valid face added for:", entry)
                            break
        

def main():
    ''' 
    Does all the work!
    '''
    
    ie = IECore() # Inference Engine Core object
    # get the network parameters for the face detection network    
    my_fd_network = get_network_information(ie, fd_xml)
    # get the network parameters for facenet
    my_facenet_network = get_network_information(ie, facenet_xml)

    # Read all of the person images from the validated faces folder. 
    validated_person_images = read_all_validated_images()
    print("number of validated images: ", len(validated_person_images))
    
    # Get all of the feature vectors from the valid person images
    valid_processed_faces = get_validated_faces(my_fd_network, my_facenet_network, validated_person_images)

    # Run the inferences on frames from the camera and make comparisons vs valid feature vectors
    run_camera(valid_processed_faces, my_fd_network, my_facenet_network)
    print(" Finished.\n")


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())

