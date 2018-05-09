# TopCoder NCS Challenge: Project template to support https://developer.movidius.com/competition

This directory contains all supporting files needed to generate `submissions.zip` file, which would then be uploaded to the TopCoder leaderboard for automatic scoring.

## Prerequisites

This code example requires that the following components are available:
1. <a href="https://developer.movidius.com/buy" target="_blank">Movidius Neural Compute Stick</a>
2. <a href="https://developer.movidius.com/start" target="_blank">Movidius Neural Compute SDK</a>

## Running the Example
~~~
mkdir -p ~/workspace
cd ~/workspace
git clone https://github.com/movidius/ncappzoo
cd ~/workspace/ncappzoo/apps/topcoder_example
make run
~~~

If everything went well, you should see an output similar to this:
~~~
Downloading training dataset...
...
Downloading provisional dataset...
...
Downloading labels file...
...
mvNCCompile v02.00, Copyright @ Movidius Ltd 2016
...
IMAGE_MEAN (108, 118, 128)
IMAGE_SCALE 0.017241379310344827
IMAGE_DIM (256, 256)
n_images = 2000
progess 100/2000 ...
progess 200/2000 ...
...
progess 1900/2000 ...
progess 2000/2000 ...
~~~

Now that you have a working framework, follow these instructions to create `submissions.zip`

1. Compile your custom trained network.
   + For Caffe based networks, rename your weights file as weights.caffemodel, deploy.prototxt as network.prototxt, and then call `make compile`.
   + For Tensorflow based networks, refer https://movidius.github.io/ncsdk/tf_compile_guidance.html.
2. Run inference on provisional dataset and generate `inferences.csv`
   + `make infer`
3. Compress all files required to make a valid submission
   + `make zip`

## Troubleshooting

If the dataset download didn't complete, you can either delete the `data` directory and rerun `make run`, or manually download the files into this directory structure.
~~~
data/
data/provisional/
data/provisional/provisional_{00001..02000}.jpg
data/training/
data/training/training_{00001..80000}.jpg
data/training_ground_truth.csv
data/provisional.csv
~~~

