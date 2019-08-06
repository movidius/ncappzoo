# AgeGenderNet
## Introduction
The [Age/GenderNet](https://github.com/opencv/open_model_zoo/blob/master/intel_models/age-gender-recognition-retail-0013/description/age-gender-recognition-retail-0013.md) network can be used for image classification. This model was trained to classify ages from 18-75. The model has 2 outputs, one for age: 'age-conv3' and another output for gender: 'prob'. 

The provided Makefile does the following

1. Downloads the IR files from the [Open Model Zoo](https://github.com/opencv/open_model_zoo)
2. Takes an image and runs the face-detection-retail-0004 sample and outputs a cropped image.
3. Takes the cropped image and runs an inference with the age-gender model using the age and gender outputs.

## Running this Example
~~~
make run
~~~

## Makefile
Provided Makefile describes various targets that help with the above mentioned tasks.

### make run
Runs a sample application with the network.

### make run_py
Runs the agenet.py python script which sends a single image to the Neural Compute Stick and receives and displays the inference results.

### make help
Shows makefile possible targets and brief descriptions. 

### make all
Makes the follow items: deps, data.

### make compile_model
Uses the network description and the trained weights files to generate an IR (intermediate representation) format file.  This file is later loaded on the Neural Compute Stick where the inferences on the network can be executed.  

### make clean
Removes all the temporary and target files that are created by the Makefile.

