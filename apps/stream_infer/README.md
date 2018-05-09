# stream_infer: A Movidius Neural Compute Stick Example

This directory contains a code example that demonstrates how to use the MVNC API to create a simple python application that pulls streaming video from a USB camera and pushes frames to the Movidius NCS, which then makes inferences based on a Convolutional Neural Network (CNN).   

## Prerequisites

This code example requires that the following hardware components are available:
1. Movidius NCS
2. USB camera that supports video for linux
3. The following multimedia plugins:
    - gstreamer1.0-libav
    - gstreamer1.0-plugins-bad-faad
    - gstreamer1.0-plugins-bad-videoparsers

Note: without the gstreamer multimedia plugins when the app runs it will just hang with a dark window.

## Running the sample

To run the example code do the following:
1. Plug the USB camera and Movidius NCS into your development computer's USB port or into a USB hub that is plugged into your development computer.
2. Run the example with the following command: python3 py_stream_infer.py

When the application is running properly you will see a window that has the camera preview with the top inference result at the bottom like this:
![](https://github.intel.com/nsmith1/sizzle/blob/master/core_tech/joule/movidius/py_stream_infer/images/py_stream_infer.jpg "stream_infer screen shot")

As you can see in this [video](https://github.intel.com/nsmith1/sizzle/blob/master/core_tech/joule/movidius/py_stream_infer/videos/py_stream_infer.mp4) the camera frames are being updated constantly as the application runs.  The Movidius NCS provides inference information on the streaming camera data as the frames are available.  For each frame that is passed to the Movidius NCS the application displays the highest scoring inference along with its percent certainty. 

## The Code
In order to make this example as clear as possible the code for it is all contained in a single source file: stream_infer.py.  To prevent obscuring the key concepts with excessive structure, the file is layed out as a series of functions with no classes declared.  The interesting sections of the stream_infer.py file are explained in this section.

---
```python
import mvnc.mvncapi as fx
```
The line above imports the Movidius Neural Compute API for use in the program.



---
```python
NETWORK_IMAGE_WIDTH = 227                     # the width of images the network requires
NETWORK_IMAGE_HEIGHT = 227                    # the height of images the network requires
NETWORK_IMAGE_FORMAT = "BGR"                  # the format of the images the network requires
NETWORK_DIRECTORY = "../../caffe/SqueezeNet/" # directory of the network 
NETWORK_STAT_TXT = "./squeezenet_stat.txt"    # stat.txt for networ
NETWORK_CATEGORIES_TXT = "./squeezenet_categories.txt" # categories.txt for network
```
The variables that start with NETWORK_ are describing the CNN that the program will be running inferences against.  These values are specific to the particular network and will likely need to be changed if you would like to use a different network.  The comments in the code explain what each value represents.  

---
```python
if __name__=="__main__":
    Gdk.init([])
    Gtk.init([])
    Gst.init([])
```
The main entry point for the program is of course denoted with if __name__=="__main__" in the code above. This is followed by initialization of Gdk, Gtk, and Gst.

---
```python
    # load means and stds from stat.txt
    with open(NETWORK_STAT_TXT, 'r') as f:
        gNetworkMean = f.readline().split()
        gNetworkStd = f.readline().split()
        for i in range(3):
            gNetworkMean[i] = 255 * float(gNetworkMean[i])
            gNetworkStd[i] = 1 / (255 * float(gNetworkStd[i]))
```
After everything is initialized the first thing the program does is read in the values from the stat.txt file specified  It's up to the developer to create the stat.txt file that is appropriate for his or her CNN.  For this example the SqueezeNet stat.txt file contains two lines with 3 numeric values separated by spaces in each line.  The first line values are the RGB mean values and the second line values are the standard deviation values for the CNN training dataset.  These are normalized to values between 0 and 255 and then stored in the global arrays gNetworkMean and gNetworkStd.  The values in these arrays will be used to preprocess the captured camera frames before handing them off to the Movidius NCS for inferences. 

---
```python
    # Load categories from categories.txt
    gNetworkCategories = []
    with open(NETWORK_CATEGORIES_TXT', 'r') as f:
        for line in f:
            cat = line.split('\n')[0]
            if cat != 'classes':
                gNetworkCategories.append(cat)
        f.close()
```
Next the categories.txt file is read from the path specified and stored in the gNetworkCategories variable.  As with the stat.txt file this file will be specific to the CNN and must be provided by the developer.  The first line of this file must have the word "classes" on a line by itself.  The following lines should be human readable names for each class that the CNN recognizes in the order that matches the category indexes that will be returned by the CNN inferences.  These text values will be referred to as each inference is made so that a human readable string can be displayed to the user.

---
```python
fx.SetGlobalOption(fx.GlobalOption.LOGLEVEL, 2)
```
The line above sets the logging level to 2 which is verbose.  Setting to 0 is no logging and 1 is errors only.

---
```python
cam_src_str = "v4l2src device=/dev/video" + CAMERA_INDEX
```
The line above sets the GStreamer camera source to the USB Camera at the index specified by CAMERA_INDEX.  0 is the first USB camera, followed by 1 etc.

---
```python
app_launch_str = "\
  videoscale ! video/x-raw, width=%s, height=%s ! \
  videoconvert ! video/x-raw, format=%s ! \
  appsink name=%s max-buffers=1 drop=true enable-last-sample=true" % (NETWORK_IMAGE_WIDTH, NETWORK_IMAGE_HEIGHT, NETWORK_IMAGE_FORMAT, GST_APP_NAME )
  
view_launch_str = "\
  queue max-size-buffers=2 leaky=downstream ! \
  xvimagesink name=%s" % GST_VIEW_NAME  
  
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
```
The app_launch_str is used in the GStreamer launch string to set up the application sink such that the images are received by the application are the width, height, and format appropriate for the CNN. 

The view_launch_str is used in the GStreamer launch string to set up the view sink that is displayed to the user as a camera preview.  The camera preview is set go to the gDrawAreaSink, which is a gtk DrawingArea instance.  This DrawingArea is set to be double buffered, and given the name to match the name in the launch string.

Finally, the entire GStreamer launch string is put into the launch variable.  This string basically defines that the specified USB camera will provide frames to the viewing (DrawingArea) sink and also to the application sink which will be used as the input to the network.

---
```python
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
```
Next the code above sets up the GTK GUI for the application which consists of a window that has a box container that holds two widgets.  The first widget is the DrawingArea that shows the camera preview, and the second widget is a label that displays the current human-readable result of the network's inference on the last camera frame (from the Movidius NCS.)

---
```python
    ncs_names = fx.EnumerateDevices()
    dev = fx.Device(ncs_names[0])
    dev.OpenDevice()
    gGraph = dev.AllocateGraph(get_graph_from_disk())
```
This code sets up the Movidius NCS for use with the network. First the ncs_names variable gets set to the names of all the Movidius NCS devices plugged into the development computer.  There may only be one in which case this will return an array of one name.  Next the dev variable is created from the name of the first device in the array.  Note that there is no error handling in this example for the case where no devices are plugged in.  In production code you would want to handle this case.  After the device is created it's opened, and finally the device is loaded with the graph for the network that the application will be using.  The get_graph_from_disk() function just reads the graph file from disk to memory.

---
```python
start_thread()
```
The start_thread() function starts two threads; one for input to the Movidius NCS and one for output from the Movidius NCS.

```python
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
```
The thread function providing input to the Movidius NCS is input_thread().  It constantly gets camera frames via the GStreamer application sink and loads them to the Movidius NCS as input to the network. The thread function handling the output of the Movidius NCS is output_thread().  It constantly gets the results (inferences) from the Movidius NCS and updates the UI to reflect them.  The call to GLib.idle_add() sets a function, update_ui(), to be called when there are no higher priority events that need to be handled.

---
```python
def input_thread():
    """ input thread function
    """
    global gRunning
    frame_number = 0
    while gRunning:
        preprocessed_image_buf = get_sample()
        if preprocessed_image_buf is not None:      
            gGraph.LoadTensor(preprocessed_image_buf ,"frame %s" % frame_number)
            frame_number=frame_number + 1

    print("Input thread terminating.")
```
The input thread runs in a loop until the gRunning boolean is set to false.  While it's running it gets an image from camera via the gstreamer pipeline then passes it to the Movidius NCS via the gGraph.LoadTensor() method.

```python
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
```
The get_sample() function above is where the camera image is brought into the application from the GStreamer pipeline.  First the sample variable is set from the 'last-sample' property of the global application sink object gGstAppSink.  Next the sample buffer is mapped for reading. Finally preprocessing is performed on the image data via the call to preprocess().  This preprocessing step is where any image adjustments required by the network can be made.  The resulting preprocessed_image_buffer is then fed into the Movidius NCS via gGraph.LoadTensor().

---
```python
def output_thread():
  """ output thread function
  for getting inference results from Movidius NCS
  running graph specific post processing of inference result
  queuing the results for main thread callbacks
  """
  global gRunning

  try:
    while gRunning:
      inference_result, user_data = gGraph.GetResult()
      gUpdateq.put((postprocess(inference_result), user_data))
  except Exception as e:
    print(e)
    pass
  print("Output thread terminating")
```
The output_thread() function above is where the inferences from the Movidius NCS are brought into the application.  It runs in a loop until the gRunning global is set to false.  Each time through the loop it tries to get a result from the Movidius NCS API via the gGraph.GetResult() method call.  As results are retrieved from the Movidius NCS they are converted to a human readable string in the postprocess() function.  This string is then put on the queue named gUpdateq.  The update_ui() function, which is called periodically from the windowing system, will pull from this queue to update the UI appropriately.

---

That's it for the code.  In summary, the Movidius is initialized with the graph file via the API's Device.AllocateGraph() function.  Then two threads are started. One pulls images from the camera via GStreamer and then pushes those images (preprocessed) to the Movidius NCS for inferencing based on the CNN graph file loaded via the API's Graph.LoadTensor() method.  The other thread pulls the inference results from the Movidius NCS via the API's Graph.GetResult() method, creates a human readable string, and pushes it to a label in the user interface for the user to read.


## Modifying the Code to Use a Different CNN
This example was written such that changing it to run inferences on a different Convolutional Neural Network is fairly easy.  In general the things that need to be changed for different CNNs are:
1. A graph file for the new CNN must be generated via the Movidius NC Toolkit (or obtained otherwise).  The API and the Toolkit both provide some sample graph files.
2. A stat.txt and categories.txt file for the CNN must be created (or obtained otherwise).
3. The graph file, stat.txt, and categories.txt files for the new CNN must be placed in an empty directory somewhere on the file system.
4. The NETWORK_DIRECTORY variable must be changed to hold the directory where the graph, stat.txt and categories.txt files are.
5. The IMAGE_WIDTH, IMAGE_HEIGHT must be changed to hold the height and width of the images that the new CNN expects.
6. The IMAGE_FORMAT variable might need to be changed if the image format the new CNN expects isn't RGB.  For instance, if it expects a gray scale image this might need to be changed to something like GRAY8.

For steps 1 and 2 above if the new CNN is not one of the provided examples you will need to refer to the Movidius NC Toolkit User Guide which provides the tools and examples to create these files.


