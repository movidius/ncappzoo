# Neural Compute Application Zoo (ncappzoo) 
[![Stable release](https://img.shields.io/badge/For_OpenVINO™_Version-2019.R2-green.svg)](https://github.com/opencv/dldt/releases/tag/2019_R1)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Welcome to the Intel<sup><sup><sup>®</sup></sup></sup> Neural Compute Application Zoo (ncappzoo). This repository is a place for any interested developers to share their projects (code and Neural Network content) that make use of the [Intel<sup><sup><sup>®</sup></sup></sup> Neural Compute Stick 2 (Intel<sup><sup><sup>®</sup></sup></sup> NCS 2)](https://software.intel.com/en-us/neural-compute-stick)  or the original [Intel<sup><sup><sup>®</sup></sup></sup> Movidius<sup><sup><sup>™</sup></sup></sup> Neural Compute Stick](https://software.intel.com/en-us/movidius-ncs) and the Deep Learning Deployment Toolkit (DLDT) portion of the [OpenVINO<sup><sup><sup>™</sup></sup></sup> Toolkit](https://software.intel.com/en-us/openvino-toolkit).
 
The ncappzoo is a community repository with many content owners and maintainers. All ncappzoo content is open source and being made available in this central location for others to download, experiment with, modify, build upon, and learn from.

## ncappzoo Quick Start
If you have an  Intel<sup><sup><sup>®</sup></sup></sup> NCS 2 (or the first generation Intel<sup><sup><sup>®</sup></sup></sup> Movidus<sup><sup><sup>™</sup></sup></sup> NCS) device and want to jump into the ncappzoo its easy!  There are only a few steps to get going with the ncappzoo projects fast.
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

- **master** branch: This is the most current branch, and the content relies on the DLDT from the OpenVINO<sup><sup><sup>™</sup></sup></sup> Toolkit.  This is the only branch that is compatible with the Intel<sup><sup><sup>®</sup></sup></sup> NCS 2 however, it is also compatible with the original Intel<sup><sup><sup>®</sup></sup></sup> Movidius<sup><sup><sup>™</sup></sup></sup> NCS device.
- **ncsdk2** branch: This branch is a legacy branch and the content relies on the NCSDK 2.x tools and APIs rather than the OpenVINO<sup><sup><sup>™</sup></sup></sup> toolkit. This branch is only compatible with the original Intel<sup><sup><sup>®</sup></sup></sup> Movidius<sup><sup><sup>™</sup></sup></sup> NCS device and is **NOT** compatile with the Intel<sup><sup><sup>®</sup></sup></sup> NCS 2 device.
- **ncsdk1** branch: This branch is a legacy branch and the content relies on the NCSDK 1.x tools and APIs rather than OpenVINO<sup><sup><sup>™</sup></sup></sup> toolkit.  This branch is only compatible with the original Intel<sup><sup><sup>®</sup></sup></sup> Movidius<sup><sup><sup>™</sup></sup></sup> NCS device and is **NOT** compatile with the Intel<sup><sup><sup>®</sup></sup></sup> NCS 2 device.

You can use the following git command to use the master branch of the repo:
```bash
git clone https://github.com/movidius/ncappzoo.git
```

## ncappzoo Compatibility Requirements

### Hardware compatibility
The projects in the ncappzoo are periodically tested on Intel<sup><sup><sup>®</sup></sup></sup> x86-64 Systems unless otherwise stated in the project's README.md file.  Although not tested on other harware platforms most projects should also work on any hardware which can run the OpenVINO Toolkit including the Raspberry Pi 3/3B/3B+/4B hardware<br><br>
The projects in the ncappzoo work on both the Intel<sup><sup><sup>®</sup></sup></sup> NCS 2 and the original Intel<sup><sup><sup>®</sup></sup></sup> Movidius NCS devices.


### Operating System Compatibility
The projects in the ncappzoo are tested and known to work on the **Ubuntu 16.04**.  These projects will likely work on other Linux based operating systems as well but they aren't tested on those unless explicitly stated in the project's README.md file and there may be some tweaks required as well.  If any specific issues are found for other OSes please submit a pull request as broad compatibility is desirable.

### OpenVINO and DLDT Compatibility
The projects in the **master branch** depend on the Deep Learning Deployment Toolkit (DLDT) portion of the OpenVINO<sup><sup><sup>™</sup></sup></sup> toolkit.  There are two flavors of the the OpenVINO<sup><sup><sup>™</sup></sup></sup> toolkit's DLDT:  
- The [Intel<sup><sup><sup>®</sup></sup></sup> Distribution of the OpenVINO<sup><sup><sup>™</sup></sup></sup> toolkit](https://software.intel.com/en-us/openvino-toolkit) is a binary installation for supported platforms.  Here are some links regarding the Intel Distribution of the OpenVINO<sup><sup><sup>™</sup></sup></sup> Toolkit and the Intel<sup><sup><sup>®</sup></sup></sup> NCS 2
  - Getting started web page: https://software.intel.com/en-us/articles/get-started-with-neural-compute-stick
  - Getting Started Video for Linux: https://youtu.be/AeEjQKKkPzg?list=PL61cFkSnEEmOF3AJvLtlDTSbwjlCP4iCs
  - OpenVINO Toolkit documentation: https://docs.openvinotoolkit.org/latest/index.html
- The [open source distribution of the OpenVINO toolkit DLDT](https://github.com/opencv/dldt).  This is the means by which the Intel NCS 2 device can be used with most single board computers on the market and is also helpful for other non-Ubuntu development systems.  Here are some links regarding the open source distribution of the OpenVINO<sup><sup><sup>™</sup></sup></sup> with the Intel<sup><sup><sup>®</sup></sup></sup> NCS 2: 
  - Applies to all target system: https://software.intel.com/en-us/articles/intel-neural-compute-stick-2-and-open-source-openvino-toolkit
  - ARMv7: https://software.intel.com/en-us/articles/ARM-sbc-and-NCS2
  - ARM64: https://software.intel.com/en-us/articles/ARM64-sbc-and-NCS2
  - Python on all: https://software.intel.com/en-us/articles/python3-sbc-and-ncs2


The projects in the ncappzoo work with both flavors of the OpenVINO<sup><sup><sup>™</sup></sup></sup> Toolkit and unless oterwise specified in a project's README.md file all projects are based on the **OpenVINO Toolkit 2019 R2 release**.

### OpenCV Compatibility
Some projects also rely on OpenCV, for these projects, the OpenCV from the OpenVINO release is the expected version.  Other versions may also work but are not tested an may require tweaks to get working.  

### Python Compatibility
The Python projects in the ncappzoo rely on python 3.5 unless otherwise stated in the project's README.md



## ncappzoo Repository Layout
The ncappzoo contains the following top-level directories.  See the README file in each of these directories or just click on the links below to explore the contents of the ncappzoo.
- **[apps](apps/README.md)** : Applications built to use the Intel Movidius NCS.  **This is a great place to start in the ncappzoo!**
- **[networks](networks/README.md)** : Scripts to download models and optimize neural networks based on any framework for use with the NCS and NCS 2.
- **[caffe](caffe/README.md)** : Scripts to download caffe models and optimize neural networks for use with the NCS and NCS 2.  Note: this is a legacy directory and new networks will be in the _networks_ directory.
- **[tensorflow](tensorflow/README.md)** : Scripts to download TensorFlow<sup><sup><sup>™</sup></sup></sup> models and optimize neural networks for use with the NCS and NCS 2.  Note: this is a legacy directory and new networks will be in the _networks_ directory.
- **[data](data/README.md)** : Data and scripts to download data for use with models and applications that use the NCS and NCS 2

The top-level directories above have subdirectories that hold project content. Each of these project subdirectories has one or more owners that assumes responsibility for it. The [OWNERS](OWNERS) file contains the mapping of subdirectory to owner. 

## Contributing to the ncappzoo
The more contributions to the ncappzoo, the more successful the community will be! We always encourage everyone with Neural Compute Stick related content to share by contributing their applications and model related work to the ncappzoo. It's easy to do, and even when contributing new content, you will be the owner and maintainer of the content.

See the [CONTRIBUTING.md](CONTRIBUTING.md) file for instructions and guidelines for contributing.

## Licensing
All content in the ncappzoo is licensed via the [MIT license](https://opensource.org/licenses/MIT) unless specifically stated otherwise in lower-level projects. Individual model and code owners maintain the copyrights for their content, but provide it to the community in accordance with the MIT License.

See the [LICENSE](LICENSE) file in the top-level directory for all licensing details, including reuse and redistribution of content in the ncappzoo repository.

