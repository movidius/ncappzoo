# live-object-detector

Detect objects in a LIVE camera feed using Intel® Movidius™ Neural Compute Stick (NCS).

## Prerequisites

This code example requires that the following components are available:
1. <a href="https://developer.movidius.com/buy" target="_blank">Movidius Neural Compute Stick</a>
2. <a href="https://developer.movidius.com/start" target="_blank">Movidius Neural Compute SDK</a>

## Running this example

~~~
mkdir -p ~/workspace
cd ~/workspace
git clone https://github.com/movidius/ncappzoo
cd ~/workspace/ncappzoo/apps/live-object-detector/
make run
~~~
 
When the application runs normally and is able to connect to the NCS device, you will see a live feed from your camera with one or more bounding boxes drawn around the detected objects. You can hit 'Q' or 'q' at any time to quit the app. You will also see the following messages being printed on the console.

~~~
I found these objects in  ( 80.62 ms ):
63.4%	9: chair: Top Left: (120, 262) Bottom Right: (477, 645)
95.1%	15: person: Top Left: (9, 318) Bottom Right: (482, 635)
~~~

## Configuring this example

This example grabs camera frames from `/dev/video0` by default; If your system has multiple cameras you can choose the required camera using the `--video` option. Below is an example:

~~~
python3 live-object-detector.py --video 1
~~~

## Customizing this example

You can use this project as a template for your custom SSD object detector app. Below are some tips to help customize the example.

1. Before attemping to customize, check if the built-in options would suffice. Run `python3 object-detector.py -h` to list all available options.
2. Steps 1, 2 and 5 are common across all Neural Compute Stick apps, so you can re-use those fuctions without modifications.
3. Step 3, 'Pre-process the images' is probably the most customizable function. As the name suggests, you can include all image pre-processing tasks in this function. Ex. if you don't want to warp the input image, just crop it before calling `skimage.transform.resize`.
4. [Future functionality] If you decide to use a different model, the `deserialize_output.ssd` function call in step 4 should be replaced by a deserializer module suitable for the chosen model.
