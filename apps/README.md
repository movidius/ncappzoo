# Application examples for the Intel Neural Compute Stick (NCS)

This directory contains subdirectories for applications that make use of the NCS.  Typically the applications here make use of one or more of the neural networks in the caffe, and/or tensorflow directories.  They are also intended to be more involved and provide more of a real world application of the networks rather than simply serving as an example of the technology.

# The Applications
The following list gives a brief description of each of the applications.

- **MultiStick_GoogLeNet:** A demo that makes use of multiple NCS sticks all executing Caffe GoogLeNet image classificaiton simultaneously demonstrating scalability. One GUI window shows inferences on a single stick and an other window uses the rest of the sticks in the system. 
- **MultiStick_TF_Inception:** Similar to MultiStick_GoogLeNet but uses TensorFlow Inception network.
- **benchmarkncs:** Runs multiple inferences on multiple neural networks within the repository and returns inference per second results for each one.  If multiple NCS devices are plugged in will give numbers for one device and for multiple.
- **birds:**
- **classifier-gui:**
- **gender_age_lbp:**
- **hello_ncs_cpp:**
- **hello_ncs_py:**
- **image-classifier:**
- **live-image-classifier:**
- **log-image-classifier:**
- **multistick_cpp:**
- **rapid-image-classifier:**
- **stream_infer:**
- **stream_ty_gn:**
- **stream_ty_gn_threaded:**
- **street_cam:**
- **street_cam_threaded:**
- **topcoder_example:**
- **video_face_matcher:**
- **video_objects:**
