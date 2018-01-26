# live-image-classifier

Perform image classification on a LIVE camera feed using deep neural networks (DNNs) on Intel® Movidius™ Neural Compute Stick (NCS). This project was used to build a battery powered, RPi based, portable inference device. You can read more about this project at the <a href="https://movidius.github.io/blog/battery-powered-dl-engine/">NCS developer blog</a>.

> This sample code doesn't necessarily need a RPi board; it can be run on any computer with a camera.

## Prerequisites

This code example requires that the following components are available:
1. <a href="https://developer.movidius.com/buy" target="_blank">Movidius Neural Compute Stick</a>
2. <a href="https://developer.movidius.com/start" target="_blank">Movidius Neural Compute SDK</a>
3. A computer with web camera

## Running this example

~~~
mkdir -p ~/workspace
cd ~/workspace
git clone https://github.com/movidius/ncappzoo
cd ~/workspace/ncappzoo/apps/live-image-classifier/
make run
~~~

When the application runs normally and is able to connect to the NCS device, you will see a live feed from your camera with a virtual box drawn over the feed. Place an item/object within this box to see inference results. You can hit 'Q' or 'q' at any time to quit the app.
