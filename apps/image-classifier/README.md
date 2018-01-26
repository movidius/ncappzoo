# image-classifier

Perform image classification using deep neural networks (DNNs) on Intel® Movidius™ Neural Compute Stick (NCS). The <a href="https://movidius.github.io/blog/ncs-image-classifier/">NCS developer blog</a> has a step by step tutorial on how to build this project, and also has a detailed explanation of the source code.

## Prerequisites

This code example requires that the following components are available:
1. <a href="https://developer.movidius.com/buy" target="_blank">Movidius Neural Compute Stick</a>
2. <a href="https://developer.movidius.com/start" target="_blank">Movidius Neural Compute SDK</a>

## Running this example

~~~
mkdir -p ~/workspace
cd ~/workspace
git clone https://github.com/movidius/ncappzoo
cd ~/workspace/ncappzoo/apps/image-classifier/
make run
~~~

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



