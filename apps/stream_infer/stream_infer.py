#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory. 

# Python script to start a USB camera and feed frames to
# the Movidius Neural Compute Stick that is loaded with a
# CNN graph file and report the inferred results

import mvnc.mvncapi as fx

import gi
gi.require_version('Gst', '1.0')
gi.require_version('Gdk', '3.0')
gi.require_version('Gtk', '3.0')
gi.require_version('GLib','2.0')
gi.require_version('GstVideo', '1.0')

from gi.repository import Gdk
from gi.repository import Gst
from gi.repository import Gtk
from gi.repository import GstVideo
from gi.repository import GLib
from gi.repository import GdkX11

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

from queue import Queue
from gi.repository import GLib
from threading import Thread

from gi.repository import Gst

import numpy

'''
NETWORK_IMAGE_WIDTH = 227       # the width of images the network requires
NETWORK_IMAGE_HEIGHT = 227      # the height of images the network requires
NETWORK_IMAGE_FORMAT = "BGR"    # the format of the images the network requires
NETWORK_DIRECTORY = "../../caffe/GenderNet/" # directory of the network this directory needs to
                                # have 3 files: "graph", "stat.txt" and "categories.txt"
NETWORK_STAT_TXT = "./gendernet_stat.txt"    # stat.txt for networ
NETWORK_CATEGORIES_TXT = "./gendernet_categories.txt" # categories.txt for network


NETWORK_IMAGE_WIDTH = 224           # the width of images the network requires
NETWORK_IMAGE_HEIGHT = 224          # the height of images the network requires
NETWORK_IMAGE_FORMAT = "BGR"        # the format of the images the network requires
NETWORK_DIRECTORY = "../../caffe/GoogLeNet/"  # directory of the network this directory needs to
                                    # have 3 files: "graph", "stat.txt" and "categories.txt"
NETWORK_STAT_TXT = "./googlenet_stat.txt"    # stat.txt for networ
NETWORK_CATEGORIES_TXT = "./googlenet_categories.txt" # categories.txt for network

'''
NETWORK_IMAGE_WIDTH = 227                     # the width of images the network requires
NETWORK_IMAGE_HEIGHT = 227                    # the height of images the network requires
NETWORK_IMAGE_FORMAT = "BGR"                  # the format of the images the network requires
NETWORK_DIRECTORY = "../../caffe/SqueezeNet/" # directory of the network 
NETWORK_STAT_TXT = "./squeezenet_stat.txt"    # stat.txt for networ
NETWORK_CATEGORIES_TXT = "./squeezenet_categories.txt" # categories.txt for network

'''
NETWORK_IMAGE_WIDTH = 227           # the width of images the network requires
NETWORK_IMAGE_HEIGHT = 227          # the height of images the network requires
NETWORK_IMAGE_FORMAT = "BGR"        # the format of the images the network requires
NETWORK_DIRECTORY = "../../caffe/AlexNet/"    # directory of the network this directory needs to
                                    # have 3 files: "graph", "stat.txt" and "categories.txt
NETWORK_STAT_TXT = "./alexnet_stat.txt"    # stat.txt for networ
NETWORK_CATEGORIES_TXT = "./alexnet_categories.txt" # categories.txt for network

'''

# The capture dimensions of the image need to be a multiple of 4 (the image will be cropped back down for inferences)
NETWORK_IMAGE_WIDTH_4 = NETWORK_IMAGE_WIDTH + ((4 - (NETWORK_IMAGE_WIDTH % 4)) % 4)
NETWORK_IMAGE_HEIGHT_4 = NETWORK_IMAGE_HEIGHT + ((4 - (NETWORK_IMAGE_HEIGHT % 4)) % 4)

GST_APP_NAME = "app"            # gstreamer sink name
GST_VIEW_NAME = "view"          # gstreamer view sink name
CAMERA_INDEX = "0"              # 0 is first usb cam, 1 the second etc.
SINK_NAME="xvimagesink"         # use for x86-64 platforms
#SINK_NAME="glimagesink"	# use for Raspian Jessie platforms

# Globals for the program
gGstAppSink = None
gIt = None
gRunning = False
gOt = None
gNetworkMean = None
gNetworkStd = None
gNetworkCategories = None
gUpdateq = Queue()
gGraph = None
gCallback = None
gResultLabel = Gtk.Label()          # label to display inferences in
gDrawAreaSink = Gtk.DrawingArea()   # DrawingArea to display camera feed in.
# end of globals for the program

# connected to the the GUI window and is called when the window is closed
def window_closed (widget, event, pipeline):
    """
    :param widget: the GTK window
    :param event:
    :param pipeline: the Gst Pipeline
    :return: nothing
    """
    widget.hide()
    pipeline.set_state(Gst.State.NULL)
    Gtk.main_quit ()


# Start the input and output worker threads for the application
def start_thread():
    """ start threads and idle handler (update_ui) for callback dispatching
    """
    global gIt, gOt, gRunning
    gRunning = True
    GLib.idle_add(update_ui) # TODO: inefficient, find a thread safe signal/event posting method
    gIt = Thread(target = input_thread)
    gIt.start()
    gOt = Thread(target = output_thread)
    gOt.start()


#Stop worker threads for the application.  Blocks until threads are terminated
def stop_thread():
    """ stop threads
    """
    global gIt, gOt, gRunning

    # Set gRunning flag to false so worker threads know to terminate
    gRunning = False;

    # Wait for worker threads to terminate.
    gIt.join()
    gOt.join()


# Called when no higher priority events are pending in the main loop.
# Will call the callback function with the data from the update queue.
def update_ui():
    """
    Dispatch callbacks with post processed inference results
        in the main thread context

    :return: running global status
    """
    global gRunning

    while not gUpdateq.empty():
        #get item from update queue
        (out, cookie) = gUpdateq.get()
        gCallback(cookie, out)
    return gRunning


# Worker thread function for input to MVNC.
# Gets a preprocessed camera sample and calls the MVNC API to do an inference on the image.
def input_thread():
    """ input thread function
    """
    global gRunning
    frame_number = 0
    while gRunning:
        preprocessed_image_buf = get_sample()
        if preprocessed_image_buf is not None:                                    # TODO: eliminate busy looping before samples are available
            #print("loading %s : %s" % (preprocessed_image_buf.shape, preprocessed_image_buf ))
            gGraph.LoadTensor(preprocessed_image_buf ,"frame %s" % frame_number)
            frame_number=frame_number + 1

    print("Input thread terminating.")

	
# Worker thread function to handle inference results from the MVNC stick
def output_thread():
  """ output thread function
  for getting inference results from Movidius NCS
  running graph specific post processing of inference result
  queuing the results for main thread callbacks
  """
  global gRunning

  try:
    while gRunning:
      try:
        inference_result, user_data = gGraph.GetResult()
        gUpdateq.put((postprocess(inference_result), user_data))
      except KeyError:
        # This error occurs when GetResult can't access the user param from the graph, we're just ignoring it for now
        #print("KeyError")
        pass
  except Exception as e:
    print(e)
    pass
  print("Output thread terminating")


# Get a sample from the camera and preprocess it so that its ready for
# to be sent to the MVNC stick to run an inference on it.
def get_sample():
    """ get a preprocessed frame to be pushed to the graph
    """
    sample = gGstAppSink.get_property('last-sample')
    if sample:
        # a sample was available from the camera via the gstreamer app sink
        buf = sample.get_buffer()
        result, info = buf.map(Gst.MapFlags.READ)
        preprocessed_image_buffer = preprocess(info.data)
        buf.unmap(info)
        del buf
        del sample
        return preprocessed_image_buffer
    return None


# Read the graph file for the network from the filesystem.
def get_graph_from_disk():
    """
    :return: the bytes that were read from disk which are the binary graph file contents
    """

    with open(NETWORK_DIRECTORY + "graph", mode='rb') as file:
        graph_blob = file.read()
    return graph_blob


# preprocess the camera images to create images that are suitable for the
# network.  Specifically resize to appropriate height and width
# and make sure the image format is correct.  This is called by the input worker
# thread function prior to passing the image the MVNC API.
def preprocess(data):
    """ preprocess a video frame
    input - in the format specified by rawinputformat() method
    output - in the format required by the graph
    """
    resize_width = NETWORK_IMAGE_WIDTH_4
    resize_height = NETWORK_IMAGE_HEIGHT_4

    buffer_data_type = numpy.dtype(numpy.uint8) # the buffer contains 8 bit unsigned ints that are the RGB Values of the image
    image_unit8_array = numpy.frombuffer(data, buffer_data_type, -1, 0) # get the input image into an array
    actual_stream_width = int(round((2*resize_width+1)/2)) # hack, rather get this from the app sink
    image_unit8_array = image_unit8_array.reshape(actual_stream_width,resize_height,3)
    image_unit8_array = image_unit8_array[0:(resize_height-1),0:(resize_width-1),0:3]    # crop to network input size
    image_float_array = image_unit8_array.astype('float32')

    #Preprocess image changing the RGB pixel values to the values the network needs
    # to do this we subtract the mean and multiply the std for each channel (R, G and B)
    # these mean and std values come from the stat.txt file that must accompany the
    # graph file for the network.
    for i in range(3):
        image_float_array[:,:,i] = (image_float_array[:,:,i] - gNetworkMean[i]) * gNetworkStd[i]

    # Finally we return the values as Float16 rather than Float32 as that is what the network expects.
    return image_float_array.astype(numpy.float16)


# post process the results from MVNC API to create a human
# readable string.
def postprocess(output):
    """ postprocess an inference result
    input - in the format produced by the graph
    output - in a human readable format
    """
    order = output.argsort()
    last = len(gNetworkCategories)-1
    text = gNetworkCategories[order[last-0]] + ' (' + '{0:.2f}'.format(output[order[last-0]]*100) + '%) '

    # to get top 5 use this code
    #for i in range(0, min(5, last+1)):
    #    text += gNetworkCategories[order[last-i]] + ' (' + '{0:.2f}'.format(output[order[last-i]]*100) + '%) '

    return text

def put_output(userobj, out):
    """ Method for receiving the (postprocessed) results
    userobj - user object passed to the FathomExpress
    out - output
    """
    global gResultLabel
    global gDrawAreaSink

    gResultLabel.set_text("%s\n" % out)


# main entry point for the program
if __name__=="__main__":
    Gdk.init([])
    Gtk.init([])
    Gst.init([])

    # Load preprocessing data for network

    # load means and stds from stat.txt
    with open(NETWORK_STAT_TXT, 'r') as f:
        gNetworkMean = f.readline().split()
        gNetworkStd = f.readline().split()
        for i in range(3):
            gNetworkMean[i] = 255 * float(gNetworkMean[i])
            gNetworkStd[i] = 1.0 / (255.0 * float(gNetworkStd[i]))

    # Load categories from categories.txt
    gNetworkCategories = []
    with open(NETWORK_CATEGORIES_TXT, 'r') as f:
        for line in f:
            cat = line.split('\n')[0]
            if cat != 'classes':
                gNetworkCategories.append(cat)
        f.close()

    fx.SetGlobalOption(fx.GlobalOption.LOGLEVEL, 2)

    # For this program we will always use the first MVNC device.
    ncs_names = fx.EnumerateDevices()
    if (len(ncs_names) < 1):
        print("Error - No NCS devices detected. Make sure your device is connected.")
        quit()


    # the camera source string for USB cameras.  They will be /dev/video0, /dev/video1, etc.
    # for this sample we will open the first camera (/dev/video0)
    cam_src_str = "v4l2src device=/dev/video" + CAMERA_INDEX

    app_launch_str = "\
        videoscale ! video/x-raw, width=%s, height=%s ! \
        videoconvert ! video/x-raw, format=%s ! \
        appsink name=%s max-buffers=1 drop=true enable-last-sample=true" % (NETWORK_IMAGE_WIDTH_4, NETWORK_IMAGE_HEIGHT_4, NETWORK_IMAGE_FORMAT, GST_APP_NAME )

    view_launch_str = "\
        queue max-size-buffers=2 leaky=downstream ! \
        %s name=%s" % (SINK_NAME, GST_VIEW_NAME)

    # a gstreamer sink that is a gtk drawing area
    # this is the camera preview display.
    gDrawAreaSink = Gtk.DrawingArea()
    gDrawAreaSink.set_double_buffered(True)
    gDrawAreaSink.name = GST_VIEW_NAME

    # build GStreamer launch string
    source2tee = "%s ! tee name=t" % cam_src_str
    tee2view   = "t. ! %s" % view_launch_str
    tee2app    = "t. ! %s" % app_launch_str
    launch     = "%s %s %s" % (source2tee, tee2view, tee2app)

    gstPipeline = Gst.parse_launch(launch)

    gGstAppSink = gstPipeline.get_by_name(GST_APP_NAME)

    # build GUI
    window = Gtk.Window()
    window.connect("delete-event", window_closed, gstPipeline)
    window.set_default_size (640, 480)
    window.set_title ("py_stream_infer")

    box = Gtk.Box()
    box.set_spacing(5)
    box.set_orientation(Gtk.Orientation.VERTICAL)
    window.add(box)

    box.pack_start(gDrawAreaSink, True, True, 0)
    gResultLabel = Gtk.Label()

    box.pack_start(gResultLabel, False, True, 0)

    window.show_all()
    window.realize()
    gstPipeline.get_by_name(GST_VIEW_NAME).set_window_handle(gDrawAreaSink.get_window().get_xid())


    # Initialize the MVNC device

    dev = fx.Device(ncs_names[0])
    dev.OpenDevice()
    gGraph = dev.AllocateGraph(get_graph_from_disk())

    # Initialize input and output threads to pass images to the
    # MVNC device and to read results from the inferences made on thos images.

    gCallback = put_output
    start_thread()

    if gstPipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        gstPipeline.set_state(Gst.State.NULL)
    else:
        Gst.debug_bin_to_dot_file (gstPipeline,Gst.DebugGraphDetails.ALL,'playing-pipeline')    # export GST_DEBUG_DUMP_DOT_DIR=/tmp/
        Gtk.main()
        Gst.debug_bin_to_dot_file (gstPipeline,Gst.DebugGraphDetails.ALL,'shutting-down-pipeline')
        gstPipeline.set_state(Gst.State.NULL)
        print("exiting main loop")
        gGraph.DeallocateGraph()
        dev.CloseDevice()
        print("mvnc device closed")
        stop_thread()
