# rapid-image-classifier: Image classification using DNNs on Intel Movidius Neural Compute Stick (NCS)

This directory contains a python3 example that shows how to sequencially classify multiple images using deep neural networks on the Intel Movidius Neural Compute Stick

## Prerequisites

This code example requires that the following components are available:
1. Movidius Neural Compute Stick
2. Movidius Neural Compute SDK
3. Python3


## Running the Example
To run the example code do the following :
1. Open a terminal and change directory to the rapid-image-classifier base directory
2. Type the following command in the terminal: make run 

When the application runs normally and is able to connect to the NCS device, the output will be similar to this:

~~~
Pre-processing images...

---- Predictions ----
cheetah, chetah, Acinonyx jubatus
African elephant, Loxodonta africana
brain coral
jellyfish
English setter
...
~~~



