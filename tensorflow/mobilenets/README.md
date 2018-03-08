# MobileNets
## On Intel® Movidius™ Neural Compute Stick (NCS)

<a href="https://arxiv.org/abs/1704.04861" target="_blank">MobileNets</a> are a class of efficient convolutional neural networks (CNNs) designed for mobile and embedded vision applications. MobileNets use depth multiplier and image size as hyper-parameters, which can be used to tweak accuracy and latency of the model during training. This ability to tweak the model allows the model builder to train a model that strikes a perfect balance between the application requirements and hardware constrains of their system.

TensorFlow™ provides a set of pre-trained models trained on <a href="http://www.image-net.org/" target="_blank">ImageNet</a>, with different combinations of depth multiplier and image size. The Makefile in this project helps convert these <a href="https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md" target="_blank">TensorFlow MobileNet models</a> to a Movidius graph file, which can be deployed on to the Intel® Movidius™ Neural Compute Stick (NCS) for inference. The <a href="https://movidius.github.io/blog/ncs-rpi3-mobilenets/" _target="blank">NCS developer blog</a> provides some benchmarking numbers related to MobileNets running on NCS and a Raspberry Pi single board computer.

## Prerequisites

This code example requires that the following components are available:
1. <a href="https://developer.movidius.com/buy" target="_blank">Movidius Neural Compute Stick</a>
2. <a href="https://developer.movidius.com/start" target="_blank">Movidius Neural Compute SDK</a>
3. <a href="https://github.com/tensorflow/tensorflow" target="_blank">TensorFlow source repo</a>
4. <a href="https://github.com/tensorflow/models" target="_blank">TensorFlow models repo</a>

## Running this example

~~~
mkdir -p ~/workspace/tensorflow

# Clone TensorFlow source and models repo
cd ~/workspace/tensorflow
git clone https://github.com/tensorflow/tensorflow
git clone https://github.com/tensorflow/models

# Clone NC App Zoo
cd ~/workspace
git clone https://github.com/movidius/ncappzoo

# Download, export, freeze and profile model
cd ~/workspace/ncappzoo/tensorflow/mobilenets/
export TF_SRC_PATH=~/workspace/tensorflow/tensorflow
export TF_MODELS_PATH=~/workspace/tensorflow/models
make
~~~

If `make` ran normally and your computer is able to connect to the NCS device, the output will be similar to this:

~~~
Downloading checkpoint files...
...
Exporting GraphDef file...
...
Freezing model for inference...
...
Profiling the model...
...
25   MobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu6     0.9   271.5   0.424
26   MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6   102.8   705.6   2.973
27   MobilenetV1/Logits/AvgPool_1a/AvgPool                 0.1   616.2   0.158
28   MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd              2.1  2028.0   0.966
29   MobilenetV1/Predictions/Softmax                       0.0    19.1   0.200
------------------------------------------------------------------------------
                                  Total inference time                   39.41
------------------------------------------------------------------------------
Generating Profile Report 'output_report.html'...
~~~

You should also see a newly created `ncappzoo/tensorflow/mobilenets/model` folder.

## Configuring this example
This example profiles MobileNet V1 with `DEPTH=1.0` and `IMGSIZE=224` by default, but you can profile MobileNet models with other depth and image size settings. Below are some example commands:

Depth multiplier = 0.25 and Image size = 128
~~~
make DEPTH=0.25 IMGSIZE=128
~~~

Depth multiplier = 1.0 and Image size = 192
~~~
make IMGSIZE=192
~~~

### Full list of options
| Depth multiplier |
| --- |
| DEPTH=1.0 |
| DEPTH=0.75 |
| DEPTH=0.50 |
| DEPTH=0.25 |

| Image size |
| --- |
| IMGSIZE=224 |
| IMGSIZE=192 |
| IMGSIZE=160 |
| IMGSIZE=128 |

### Compiling and validating the models
The Makefile only runs NCS <a href="https://movidius.github.io/ncsdk/tools/profile.html" target="_blank">profiler</a> by default, but you can also compile and validate the model using the <a href="https://movidius.github.io/ncsdk/tools/compile.html" target="_blank">compile</a> and <a href="https://movidius.github.io/ncsdk/tools/check.html" target="_blank">check</a> targets. Below are some example commands:

Validate MobileNet_v1_0.75_160
~~~
make check DEPTH=0.75 IMGSIZE=160
~~~

Compile MobileNet_v1_0.50_128
~~~
make compile DEPTH=0.50 IMGSIZE=128
~~~

> NOTE: NCS profiler generates a Movidius graph file, so if you have already run the profiler on a specific model, there is no reason to run the compiler for the same model.

## Troubleshooting

~~~
Makefile:31: *** TF_MODELS_PATH is not defined. Run `export TF_MODELS_PATH=path/to/your/tensorflow/models/repo`.  Stop.
~~~
* Make sure TF_MODELS_PATH is pointing to your tensorflow models directory.

~~~
Makefile:46: *** TF_SRC_PATH is not defined. Run `export TF_SRC_PATH=path/to/your/tensorflow/source/repo`.  Stop.
~~~
* Make sure TF_SRC_PATH is pointing to your tensorflow source directory.

