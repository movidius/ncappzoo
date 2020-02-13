---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
title: ncappzoo - Home
nav_order: 1
---


# Neural Compute Application Zoo (ncappzoo) 
[![Stable release](https://img.shields.io/badge/For_OpenVINO™_Version-2020.1-green.svg)](https://github.com/opencv/dldt/releases/tag/2020.1)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) 


Welcome to the Neural Compute Application Zoo (ncappzoo). This repository is a place for any interested developers to share their projects (code and Neural Network content) that make use of the [Intel&reg; Neural Compute Stick 2 (Intel&reg; NCS 2)](https://software.intel.com/en-us/neural-compute-stick)  or the original [Intel&reg; Movidius&trade; Neural Compute Stick](https://software.intel.com/en-us/movidius-ncs) and the Deep Learning Deployment Toolkit (DLDT) portion of the [OpenVINO&trade; Toolkit](https://software.intel.com/en-us/openvino-toolkit).
 
The ncappzoo is a community repository with many content owners and maintainers. All ncappzoo content is open source and being made available in this central location for others to download, experiment with, modify, build upon, and learn from.

Find the repository at [GitHub](https://github.com/movidius/ncappzoo).

## ncappzoo Quick Start
If you have an  Intel&reg; NCS 2 (or the first generation Intel&reg; Movidus&trade; NCS) device and want to jump into the ncappzoo its easy!  There are only a few steps to get going with the ncappzoo projects fast.
1. clone the repo with the following command
```bash
git clone https://github.com/movidius/ncappzoo.git
```
2. Explore apps by opening a terminal window navigating to any directory under **ncappzoo/apps** and execute this command
```bash
make run
```
3. Explore the neural networks by navigating to any network directory under **ncappzoo/networks**, **ncappzoo/caffe**, or **ncappzoo/tensorflow** and execute the same command
```bash
make run
```
Thats it! All of the network and app directories have simple consistant makefiles. To see other make targets supported from these directories just execute this command 
```bash
make help
```


## ncappzoo Repository Branches
There are three branches in the repository; their discriptions are below.  **The master branch is the one most developers will want.**  The others are provided only for legacy reasons.

- **master** branch: This is the most current branch, and the content relies on the DLDT from the OpenVINO&trade; Toolkit.  This is the only branch that is compatible with the Intel&reg; NCS 2 however, it is also compatible with the original Intel&reg; Movidius&trade; NCS device.
- **ncsdk2** branch: This branch is a legacy branch and the content relies on the NCSDK 2.x tools and APIs rather than the OpenVINO&trade; toolkit. This branch is only compatible with the original Intel&reg; Movidius&trade; NCS device and is **NOT** compatile with the Intel&reg; NCS 2 device.
- **ncsdk1** branch: This branch is a legacy branch and the content relies on the NCSDK 1.x tools and APIs rather than OpenVINO&trade; toolkit.  This branch is only compatible with the original Intel&reg; Movidius&trade; NCS device and is **NOT** compatile with the Intel&reg; NCS 2 device.

You can use the following git command to use the master branch of the repo:
```bash
git clone https://github.com/movidius/ncappzoo.git
```

## ncappzoo Compatibility Requirements

### Hardware compatibility
The projects in the ncappzoo are periodically tested on Intel&reg; x86-64 Systems unless otherwise stated in the project's README.md file.  Although not tested on other harware platforms most projects should also work on any hardware which can run the OpenVINO Toolkit including the Raspberry Pi 3/3B/3B+/4B hardware<br><br>
The projects in the ncappzoo work on both the Intel&reg; NCS 2 and the original Intel&reg; Movidius NCS devices.


### Operating System Compatibility
The projects in the ncappzoo are tested and known to work on the **Ubuntu 16.04**.  These projects will likely work on other Linux based operating systems as well but they aren't tested on those unless explicitly stated in the project's README.md file and there may be some tweaks required as well.  If any specific issues are found for other OSes please submit a pull request as broad compatibility is desirable.

### OpenVINO and DLDT Compatibility
The projects in the **master branch** depend on the Deep Learning Deployment Toolkit (DLDT) portion of the OpenVINO&trade; toolkit.  There are two flavors of the the OpenVINO&trade; toolkit's DLDT:  
- The [Intel&reg; Distribution of the OpenVINO&trade; toolkit](https://software.intel.com/en-us/openvino-toolkit) is a binary installation for supported platforms.  Here are some links regarding the Intel Distribution of the OpenVINO&trade; Toolkit and the Intel&reg; NCS 2
  - Getting started web page: https://software.intel.com/en-us/articles/get-started-with-neural-compute-stick
  - Getting Started Video for Linux: https://youtu.be/AeEjQKKkPzg?list=PL61cFkSnEEmOF3AJvLtlDTSbwjlCP4iCs
  - OpenVINO Toolkit documentation: https://docs.openvinotoolkit.org/latest/index.html
- The [open source distribution of the OpenVINO toolkit DLDT](https://github.com/opencv/dldt).  This is the means by which the Intel NCS 2 device can be used with most single board computers on the market and is also helpful for other non-Ubuntu development systems.  Here are some links regarding the open source distribution of the OpenVINO&trade; with the Intel&reg; NCS 2: 
  - Applies to all target system: https://software.intel.com/en-us/articles/intel-neural-compute-stick-2-and-open-source-openvino-toolkit
  - ARMv7: https://software.intel.com/en-us/articles/ARM-sbc-and-NCS2
  - ARM64: https://software.intel.com/en-us/articles/ARM64-sbc-and-NCS2
  - Python on all: https://software.intel.com/en-us/articles/python3-sbc-and-ncs2


The projects in the ncappzoo work with both flavors of the OpenVINO&trade; Toolkit and unless oterwise specified in a project's README.md file all projects are based on the **OpenVINO Toolkit 2019 R3 release**.

### OpenCV Compatibility
Some projects also rely on OpenCV, for these projects, the OpenCV from the OpenVINO release is the expected version.  Other versions may also work but are not tested an may require tweaks to get working.  

### Python Compatibility
The Python projects in the ncappzoo rely on python 3.5 unless otherwise stated in the project's README.md

### Raspberry Pi Compatibility
The ncappzoo is compatible with the Raspberry Pi 3 B+ and the Raspberry Pi 4. Some additional configuration steps are required:

#### Intel&reg; Distribution of OpenVINO&trade; for Raspbian* OS
The **Intel&reg; Distribution of OpenVINO&trade; toolkit for Raspbian OS** does not include the Model Optimizer. To use the ncappzoo, you must clone the open source version of the OpenVINO&trade; Deep Learning Development Toolkit (DLDT) and use that version of the Model Optimizer. Clone the repository, install dependencies for TensorFlow* and Caffe*, and set up your **PATH** and **PYTHONPATH** variables to point to the Model Optimizer:
```
cd ~
git clone https://github.com/opencv/dldt.git
cd dldt/model-optimizer
pip3 install -r requirements_tf.txt
pip3 install -r requirements_caffe.txt
export PATH=~/dldt/model-optimizer:$PATH
export PYTHONPATH=~/dldt/model-optmizer:$PYTHONPATH
```
#### Open Source OpenVINO&trade; Deep Learning Development Toolkit (DLDT)
To setup the open source version of OpenVINO&trade; with your Raspberry Pi, add to the PATH, PYTHONPATH, and LD_LIBRARY_PATH environment variables the location of the build Inference Engine libraries and Python API.

## ncappzoo Repository Layout
The ncappzoo contains the following top-level directories.  See the README file in each of these directories or just click on the links below to explore the contents of the ncappzoo.
- **[apps](./apps/)** : Applications built to use the Intel Movidius NCS.  **This is a great place to start in the ncappzoo!**
- **[networks](./networks/)** : Scripts to download models and optimize neural networks based on any framework for use with the NCS and NCS 2.
- **[caffe](https://github.com/movidius/ncappzoo/tree/master/caffe/README.md)** : Scripts to download caffe models and optimize neural networks for use with the NCS and NCS 2.  Note: this is a legacy directory and new networks will be in the _networks_ directory.
- **[tensorflow](https://github.com/movidius/ncappzoo/tree/master/tensorflow/README.md)** : Scripts to download TensorFlow&trade; models and optimize neural networks for use with the NCS and NCS 2.  Note: this is a legacy directory and new networks will be in the _networks_ directory.
- **[data](https://github.com/movidius/ncappzoo/tree/master/data/README.md)** : Data and scripts to download data for use with models and applications that use the NCS and NCS 2

The top-level directories above have subdirectories that hold project content. Each of these project subdirectories has one or more owners that assumes responsibility for it. The [OWNERS](https://github.com/movidius/ncappzoo/tree/master/OWNERS) file contains the mapping of subdirectory to owner. 

## Contributing to the ncappzoo
The _ncappzoo_ is meant to explore and teach features available for the Intel&reg; Movidius&trade; Neural Compute Stick and Intel&reg; Neural Compute Stick 2 with the Intel&reg; OpenVINO&trade; toolkit. The more contributions to the ncappzoo, the more successful this community will be! We always encourage everyone with Neural Compute Stick related content to share by contributing their applications and model-related work to the ncappzoo. It's easy to do, and even when contributing new content you will be the owner and maintainer of the content.

If your inclusion is an opportunity to explore a new idea in computer vision, add as much documentation about the functionality and your process in creating your app or network, including smartly commenting your code. This will give others - and you! - a chance to learn from your addition. Your addition will help grow our community and improve all of our AI and computer vision skills. Most importantly, the insights you get from releasing your app into the wild here will only help you down the line if you ever want to commercialize your idea. As always, your work in the _ncappzoo_ should be properly attributed so that its ownership will always be managed by you and those you grant additional rights to.

See the [CONTRIBUTING.md](https://github.com/movidius/ncappzoo/tree/master/CONTRIBUTING.md) file for instructions and guidelines for contributing.

## Licensing
All content in the ncappzoo is licensed via the [MIT license](https://opensource.org/licenses/MIT) unless specifically stated otherwise in lower-level projects. Individual model and code owners maintain the copyrights for their content, but provide it to the community in accordance with the MIT License.

See the [LICENSE](https://github.com/movidius/ncappzoo/tree/master/LICENSE) file in the top-level directory for all licensing details, including reuse and redistribution of content in the ncappzoo repository.

