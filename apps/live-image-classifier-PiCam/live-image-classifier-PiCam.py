#!/usr/bin/python3

# ****************************************************************************
# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.
# ****************************************************************************

# Perform inference on a LIVE Pi camera feed using DNNs on
# Intel® Movidius™ Neural Compute Stick (NCS)

import os
import io
import cv2
import sys
import numpy
import ntpath
import argparse
import skimage.io
import skimage.transform
from picamera.array import PiRGBArray
import picamera

import time
import mvnc.mvncapi as mvnc

# Variable to store commandline arguments
ARGS                    = None

# OpenCV object for video capture
cam 					= None
rawCapture = None
# ---- Step 1: Open the enumerated device and get a handle to it -------------

def open_ncs_device():

    # Look for enumerated NCS device(s); quit program if none found.
    devices = mvnc.enumerate_devices()
    if len( devices ) == 0:
        print( "No devices found" )
        quit()

    # Get a handle to the first enumerated device and open it
    device = mvnc.Device( devices[0] )
    device.open()

    return device

# ---- Step 2: Load a graph file onto the NCS device -------------------------

def load_graph( device ):

    # Read the graph file into a buffer
    with open( ARGS.graph, mode='rb' ) as f:
        blob = f.read()

    # Load the graph buffer into the NCS
    graph = mvnc.Graph( ARGS.graph )
        # Set up fifos
    fifo_in, fifo_out = graph.allocate_with_fifos(device, blob)

    return graph, fifo_in, fifo_out

# ---- Step 3: Pre-process the images ----------------------------------------

def pre_process_image(frame):

    height, width, channels = frame.shape

    # Extract/crop a section of the frame and resize it
    x1 = int( width / 3 )
    y1 = int( height / 4 )
    x2 = int( width * 2 / 3 )
    y2 = int( height * 3 / 4 )

    cv2.rectangle( frame, ( x1, y1 ) , ( x2, y2 ), ( 0, 255, 0 ), 2 )
    img = frame[ y1 : y2, x1 : x2 ]
    # Resize image [Image size if defined by choosen network, during training]
    img = cv2.resize( img, tuple( ARGS.dim ) )
    # Convert RGB to BGR [skimage reads image in RGB, but Caffe uses BGR]
    if( ARGS.colormode == "BGR" ):
        img = img[:, :, ::-1]
    #print(type(img))
    # Mean subtraction & scaling [A common technique used to center the data]
    img = ( img - ARGS.mean ) * ARGS.scale
    return img, frame

# ---- Step 4: Read & print inference results from the NCS -------------------

def infer_image( graph, img, frame, fifo_in, fifo_out ):

    # Load the labels file
    labels =[ line.rstrip('\n') for line in
                   open( ARGS.labels ) if line != 'classes\n']

    # Load the image as a half-precision floating point array
    graph.queue_inference_with_fifo_elem( fifo_in, fifo_out, img.astype(numpy.float32), None )

    # Get the results from NCS
    output, userobj = fifo_out.read_elem()

    # Find the index of highest confidence
    top_prediction = output.argmax()
    h,w,ch = frame.shape
    x2 = int(w*2/3)
    y1 = int(h/4)

    # Get execution time
    inference_time = graph.get_option( mvnc.GraphOption.RO_TIME_TAKEN )
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = ( "I am %3.1f%%" % (100.0 * output[top_prediction] ) + " confident"
            + " you are " + labels[top_prediction])

    cv2.putText(frame,text,(x2-40,y1+10),font,0.45,(0,0,255),2)

    # If a display is available, show the image on which inference was performed
    if 'DISPLAY' in os.environ:
        title = 'NCS live inference on Age Classification'
        #frame = cv2.flip(frame, 1)
        cv2.namedWindow(title, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(title,click_event)
        cv2.imshow( 'NCS live inference on Age Classification', frame )

# ---- Step 5: Close/clean up fifos, graph, and device -------------------------

def clean_up(device, graph, fifo_in, fifo_out):
    fifo_in.destroy()
    fifo_out.destroy()
    graph.destroy()
    device.close()
    device.destroy()
    cam.release()
    cv2.destroyAllWindows()


# ---- Step 6: Enabling touch facility on RPi boards with LCD Screens. can be disabled, if necessary -------------------------

request_to_exit = False
def click_event(event,x,y,flags,param):
    global request_to_exit
    request_to_exit = True

# ---- Main function (entry point for this script ) --------------------------

def main():

    device = open_ncs_device()
    graph, fifo_in, fifo_out  = load_graph( device )

    while not request_to_exit:

        for frame in cam.capture_continuous(rawCapture, format = "bgr", use_video_port = True):
            image = frame.array
            img,frame = pre_process_image(image)
            infer_image( graph, img, frame, fifo_in, fifo_out )

        # Display the frame for 5ms, and close the window so that the next frame
        # can be displayed. Close the window if 'q' or 'Q' is pressed.
            if( cv2.waitKey( 1 ) & 0xFF == ord( 'q' ) ):
                break
            rawCapture.truncate(0)

            if request_to_exit:
                break

    clean_up(device, graph, fifo_in, fifo_out)

# ---- Define 'main' function as the entry point for this script -------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                         description="Image classifier using \
                         Intel® Movidius™ Neural Compute Stick." )

    parser.add_argument( '-g', '--graph', type=str,
                         default='../../caffe/GenderNet/graph',
                         help="Absolute path to the neural network graph file." )

    parser.add_argument( '-l', '--labels', type=str,
                         default='../../data/age_gender/gender_categories.txt',
                         help="Absolute path to labels file." )

    parser.add_argument( '-M', '--mean', type=float,
                         nargs='+',
                         default=[78.42633776, 87.76891437, 114.89584775],
                         help="',' delimited floating point values for image mean." )

    parser.add_argument( '-S', '--scale', type=float,
                         default=1,
                         help="Absolute path to labels file." )

    parser.add_argument( '-D', '--dim', type=int,
                         nargs='+',
                         default=[227, 227],
                         help="Image dimensions. ex. -D 224 224" )

    parser.add_argument( '-c', '--colormode', type=str,
                         default="RGB",
                         help="RGB vs BGR color sequence. \
                               Defined during model training." )

    parser.add_argument( '-v', '--video', type=int,
                         default=0,
                         help="Index of your computer's V4L2 video device. \
                               ex. 0 for /dev/video0" )

    ARGS = parser.parse_args()
    # Construct (open) the Pi camera
    cam = picamera.PiCamera()
    cam.resolution = (720,576)
    rawCapture = PiRGBArray(cam)


    main()

# ==== End of file ===========================================================
