# Application examples for the Intel Neural Compute Stick (NCS)

This directory contains subdirectories for applications that make use of the NCS.  Typically the applications here make use of one or more of the neural networks in the caffe, and/or tensorflow directories.  They are also intended to be more involved and provide more of a real world application of the networks rather than simply serving as an example of the technology.

# The Applications
The following list gives a brief description of each of the applications.

- **MultiStick_GoogLeNet:** A Python demo that makes use of multiple NCS sticks all executing Caffe GoogLeNet image classificaiton simultaneously demonstrating scalability. One GUI window shows inferences on a single stick and an other window uses the rest of the sticks in the system. 
- **MultiStick_TF_Inception:** Similar to MultiStick_GoogLeNet but uses TensorFlow Inception network.
- **benchmarkncs:** Runs multiple inferences on multiple neural networks within the repository and returns inference per second results for each one.  If multiple NCS devices are plugged in will give numbers for one device and for multiple.
- **birds:** Identifies birds in photos by using Yolo Tiny to identify birds in general and then GoogLeNet to further classify them.  Images are displayed in GUI with boxes and labels around the birds found.
- **classifier-gui:** Presents the user with a GUI to select the neural network along with the image to classify and presents results. 
- **gender_age_lbp:** Uses the AgeNet and GenderNet neural networks to predict age and gender of people in a live camera feed.  The camera feed is displayed with a box overlayed around the faces and a label for age and gender of the person.  The face detection is done with OpenCV.
- **hello_ncs_cpp:** Originated as sample in the ncsdk this application shows how to open and close the NCS stick in a C++ applicaiton.
- **hello_ncs_py:** Originated as sample in the ncsdk this application shows how to open and close the NCS stick in a Python applicaiton.
- **image-classifier:** Project that accompanies the blog https://movidius.github.io/blog/ncs-image-classifier/
- **live-image-classifier:** Performs image classification on a live camera feed.  This project was used to build a battery powered, RPi based, portable inference device (although RPi isn't required.)  You can read more about this project at this NCS developer blog https://movidius.github.io/blog/battery-powered-dl-engine/
- **log-image-classifier:** Application logs results of an image classifier into a comma-separated values (CSV) file. Run inferences sequentially (and recursively) on all images within a folder.
- **multistick_cpp:** Very straightforward application that does image classification on two different networks within the same C++ application using two NCS devices.
- **rapid-image-classifier:** Performs image classification on a large number of images. This sample code was used to validate a Dogs vs Cats classifier built using a customized version of GoogLeNet. You can read more about this project (and a step-by-step guide) here https://movidius.github.io/blog/deploying-custom-caffe-models/.
- **stream_infer:** Originated as a sample in the ncsdk this Python program uses gstreamer to grab camera frames and do inferences on them. 
- **stream_ty_gn:** This GUI application shows a camera stream and sends each frame to Tiny Yolo for object detection and then crops each object and sends that to GoogLeNet for further classification. This application requires two NCS devices, one for each network.
- **stream_ty_gn_threaded:** Similar to stream_ty_gn but uses threading for better performance. 
- **street_cam:** Processes a video file (presumably produced by a street camera but not necessarily) and overlays boxes and labels around the objects detected.  The Tiny Yolo network is used for initial object detection and GoogLeNet is used for furhter classification.  This requires two NCS devices.
- **street_cam_threaded:** Similar to street_cam but uses threading for better performance.  This application will make use of as many NCS devices as are availble for the GoogLeNet classifications.
- **topcoder_example:** Contains all supporting files needed to generate submissions.zip file, which would then be uploaded to the TopCoder leaderboard for automatic scoring of the NCS competition described here: https://developer.movidius.com/competition.
- **video_face_matcher:** Uses the tensorflow/facenet network to identify faces in a camera video stream.  A single face image is used as the key and when a face in the video stream matches the key, a green frame is overlayed on the video feed.
- **video_objects:** Uses the caffe/SSD_MobileNet network to find and identify objects in a video file and displays the results in a GUI.
