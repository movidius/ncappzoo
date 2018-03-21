# Resnet_v1
## On Intel® Movidius™ Neural Compute Stick (NCS)

TensorFlow™ provides different versions of pre-trained inception models trained on <a href="http://www.image-net.org/" target="_blank">ImageNet</a>. The Makefile in this project helps convert these <a href="https://github.com/tensorflow/models/tree/master/research/slim#Pretrained" target="_blank">TensorFlow Inception models</a> to a Movidius graph file, which can be deployed on to the Intel® Movidius™ Neural Compute Stick (NCS) for inference.

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
cd ~/workspace/ncappzoo/tensorflow/inception/
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
107   InceptionV3/Logits/AvgPool_1a_8x8/AvgPool             0.3   998.1   0.254
108   InceptionV3/Logits/Conv2d_1c_1x1/BiasAdd              4.1  2080.0   1.883
109   InceptionV3/Predictions/Softmax                       0.0    18.9   0.202
-------------------------------------------------------------------------------
                                   Total inference time                  323.57
-------------------------------------------------------------------------------
Generating Profile Report 'output_report.html'...
Movidius graph generated! You can run inferences using ncappzoo/apps/image-classifier project.
~~~

## Configuring this example
This example profiles ResNet_v1_50 by default, but you can profile other inception versions as well. Below are some example commands:

ResNet_v1_101
~~~
make SUBVERSION=101
~~~

ResNet_v1_152
~~~
make SUBVERSION=152
~~~

### Full list of options
| ResNet_v1 subversions |
| --- |
| SUBVERSION=50 |
| SUBVERSION=101 |
| SUBVERSION=152 |

### Compiling and validating the models
The Makefile only runs NCS <a href="https://movidius.github.io/ncsdk/tools/profile.html" target="_blank">profiler</a> by default, but you can also compile and validate the model using the <a href="https://movidius.github.io/ncsdk/tools/compile.html" target="_blank">compile</a> and <a href="https://movidius.github.io/ncsdk/tools/check.html" target="_blank">check</a> targets. Below are some example commands:

Validate ResNet_v1_101 
~~~
make check SUBVERSION=101
~~~

Compile ResNet_v1_152
~~~
make compile SUBVERSION=152
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

