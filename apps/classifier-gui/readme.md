# classifier GUI: Image classification using DNNs on Intel Movidius Neural Compute Stick (NCS)

This directory contains a python3 example that shows how to classify images using deep neural networks on the Intel Movidius Neural Compute Stick using a simple GUI

## Prerequisites

This code example requires that the following components are available:
1. Movidius Neural Compute Stick
2. Movidius Neural Compute SDK
3. Python3
4. Install Python Image Library imagetk with this command:
   sudo apt-get install -y python3-pil.imagetk


## Running the Example
To run the example code do the following :
1. Open a terminal and change directory to the classifier-gui base directory
2. Type the following command in the terminal: make run 

When the application runs normally and is able to connect to the NCS device, the output will be similar to this:

~~~
------- predictions --------
prediction 1 is n02123159 tiger cat
prediction 2 is n02124075 Egyptian cat
prediction 3 is n02113023 Pembroke, Pembroke Welsh corgi
prediction 4 is n02127052 lynx, catamount
prediction 5 is n02971356 carton
~~~

You should also see the image on which inference was performed.



