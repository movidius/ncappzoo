#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.


from mvnc import mvncapi as mvnc
import numpy
import cv2
import os
import sys
from typing import List


NETWORK_IMAGE_DIMENSIONS = (28,28)
IMAGE_PATH = '../../data/digit_images'

def do_initialize() -> (mvnc.Device, mvnc.Graph):
    """Creates and opens the Neural Compute device and c
    reates a graph that can execute inferences on it.

    Returns
    -------
    device : mvnc.Device
        The opened device.  Will be None if couldn't open Device.
    graph : mvnc.Graph
        The allocated graph to use for inferences.  Will be None if couldn't allocate graph
    """
    # ***************************************************************
    # Get a list of ALL the sticks that are plugged in
    # ***************************************************************
    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
            print('Error - No devices found')
            return (None, None)

    # ***************************************************************
    # Pick the first stick to run the network
    # ***************************************************************
    device = mvnc.Device(devices[0])

    # ***************************************************************
    # Open the NCS
    # ***************************************************************
    device.OpenDevice()

    filefolder = os.path.dirname(os.path.realpath(__file__))
    graph_filename = filefolder + '/mnist_inference.graph'

    # Load graph file
    try :
        with open(graph_filename, mode='rb') as f:
            in_memory_graph = f.read()
    except :
        print ("Error reading graph file: " + graph_filename)

    graph = device.AllocateGraph(in_memory_graph)

    return device, graph


def do_inference(graph: mvnc.Graph, image_filename: str, number_results : int = 5) -> (List[str], List[numpy.float16]) :
    """ executes one inference which will determine the top classifications for an image file.

    Parameters
    ----------
    graph : Graph
        The graph to use for the inference.  This should be initialize prior to calling
    image_filename : string
        The filename (full or relative path) to use as the input for the inference.
    number_results : int
        The number of results to return, defaults to 5

    Returns
    -------
    labels : List[str]
        The top labels for the inference.  labels[i] corresponds to probabilities[i]
    probabilities: List[numpy.float16]
        The top probabilities for the inference. probabilities[i] corresponds to labels[i]
    """

    # text labels for each of the possible classfications
    labels=[ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


    # Load image from disk and preprocess it to prepare it for the network
    # assuming we are reading a .jpg or .png color image so need to convert it
    # single channel gray scale image for mnist network.
    # Then resize the image to the size of image the network was trained with.
    # Next convert image to floating point format and normalize
    # so each pixel is a value between 0.0 and 1.0
    image_for_inference = cv2.imread(image_filename)
    image_for_inference = cv2.cvtColor(image_for_inference, cv2.COLOR_BGR2GRAY)
    image_for_inference=cv2.resize(image_for_inference, NETWORK_IMAGE_DIMENSIONS)
    image_for_inference = image_for_inference.astype(numpy.float32)
    image_for_inference[:] = ((image_for_inference[:] )*(1.0/255.0))

    # Start the inference by sending to the device/graph
    graph.LoadTensor(image_for_inference.astype(numpy.float16), None)

    # Get the result from the device/graph.  userobj should be the
    # same value that was passed in LoadTensor above.
    output, userobj = graph.GetResult()

    # sort indices in order of highest probabilities
    five_highest_indices = (-output).argsort()[:number_results]

    # get the labels and probabilities for the top results from the inference
    inference_labels = []
    inference_probabilities = []

    for index in range(0, number_results):
        inference_probabilities.append(str(output[five_highest_indices[index]]))
        inference_labels.append(labels[five_highest_indices[index]])

    return inference_labels, inference_probabilities


def do_cleanup(device: mvnc.Device, graph: mvnc.Graph) -> None:
    """Cleans up the NCAPI resources.

    Parameters
    ----------
    device : mvncapi.Device
             Device instance that was initialized in the do_initialize method
    graph : mvncapi.Graph
            Graph instance that was initialized in the do_initialize method

    Returns
    -------
    None

    """
    graph.DeallocateGraph()
    device.CloseDevice()


def show_inference_results(image_filename: str, infer_labels: List[str],
                           infer_probabilities: List[numpy.float16]) -> None:
    """Print out results of a single inference to the console.

    Parameters
    ----------
    image_filename : str
                     The name of the image file for the inference
    infer_labels : List[str]
                   The resulting labels from the inference in order
                   of most likely to least likely.
                   Must be the same number of items as infer_probabilities.
    infer_probabilities : List[numpy.float16]
                          The resulting probabilities from the inference
                          in order of most likely to least likely.
                          Must be the same number of items as infer_labels

    Returns
    -------
    None
    """
    print("\n")
    print('-----------------------------------------------------------')
    print("Inference for " + os.path.basename(image_filename) + " ---> " + "'" + infer_labels[0]+ "'")
    print('')
    print('Top results from most certain to least:')
    num_results = len(infer_labels)
    for index in range(0, num_results):
        one_prediction = '  certainty ' + str(infer_probabilities[index]) + ' --> ' + "'" + infer_labels[index]+ "'"
        print(one_prediction)

    print('-----------------------------------------------------------')


def main():
    """ Main function, return an int for program return value

    Opens device, reads graph file, runs inferences on files in digit_images
    subdirectory, prints results, closes device
    """

    # get list of all the .png files in the image directory
    image_name_list = os.listdir(IMAGE_PATH)

    # filter out non .png files
    image_name_list = [IMAGE_PATH + '/' + a_filename for a_filename in image_name_list if a_filename.endswith('.png')]

    if (len(image_name_list) < 1):
        # no images to show
        print('No .png files in ' + IMAGE_PATH)
        return 1


    # initialize the neural compute device via the NCAPI
    device, graph = do_initialize()

    if (device == None or graph == None):
        print ("Could not initialize device.")
        quit(1)


    # lopt through all the input images and run inferences and show results
    for index in range(0, len(image_name_list)):
        infer_labels, infer_probabilities = do_inference(
            graph, image_name_list[index], 5)
        show_inference_results(image_name_list[index], infer_labels,
                               infer_probabilities)

    # clean up the NCAPI devices
    do_cleanup(device, graph)


if __name__ == "__main__":
    sys.exit(main())

