# object-detector

Detect objects in images using Single Shot Multibox Detectors (SSD) or Tiny Yolo on Intel® Movidius™ Neural Compute Stick (NCS).

## Prerequisites

This code example requires that the following components are available:
1. <a href="https://developer.movidius.com/buy" target="_blank">Movidius Neural Compute Stick</a>
2. <a href="https://developer.movidius.com/start" target="_blank">Movidius Neural Compute SDK</a>

## Running this example

~~~
mkdir -p ~/workspace
cd ~/workspace
git clone https://github.com/movidius/ncappzoo
cd ~/workspace/ncappzoo/apps/object-detector/
make run
~~~
 
When the application runs normally and is able to connect to the NCS device, the output will be similar to this:

~~~
==============================================================
I found these objects in pic_064.jpg
Execution time: 80.40859ms
--------------------------------------------------------------
99.7%	12: dog: Top Left: (46, 52) Bottom Right: (169, 182)
==============================================================
~~~

You should also see the image with a bounding box around the detected object.

## Configuring this example

This example performs an inference on `ncappzoo/data/images/pic_064.jpg` by default; you can supply your own image using the `--image` options. Below is an example:

~~~
python3 object-detector.py --image ../../data/images/pic_053.jpg
~~~

## Customizing this example

You can use this project as a template for your custom SSD/TinyYolo object detector app. Below are some tips to help customize the example.

1. Before attemping to customize, check if the built-in options would suffice. Run `python3 object-detector.py -h` to list all available options.
2. Steps 1, 2 and 5 are common across all Neural Compute Stick apps, so you can re-use those fuctions without modifications.
3. Step 3, 'Pre-process the images' is probably the most customizable function. As the name suggests, you can include all image pre-processing tasks in this function. Ex. if you don't want to warp the input image, just crop it before calling `skimage.transform.resize`.
4. [Future functionality] If you decide to use a different model, the `deserialize_output.ssd` function call in step 4 should be replaced by a deserializer module suitable for the chosen model.
