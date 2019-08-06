# Contributing to the Neural Compute App Zoo (NC App Zoo)

We are glad you want to contribute with a model and/or application for the ncappzoo. This is a great way to help the developer community. We encourage submissions from any and all developers interested in the  Intel<sup><sup><sup>®</sup></sup></sup> NCS 2 or the original Intel<sup><sup><sup>®</sup></sup></sup> Movidius<sup><sup><sup>™</sup></sup></sup> NCS.

## Submitting a Pull Request
To contribute a new app or neural network for the ncappzoo, fork the repository and add (commit) any of the following new subdirectories containing your content (your content could contain any or all of the following):
- apps/(new app name)
- networks/(new network name)
 
Note: the caffe, and tensorflow directories are for legacy comptibility and new networks should be placed in the networks directory


Your application code goes under **apps**, any neural network that you have been working with for the NCS/NCS 2 will go under **networks**.

After you have committed changes to your fork of the ncappzoo (always commit using the --signoff option as described in the Contribution Licensing section below), create a pull request for the new directories to be pulled into this repo (the ncappzoo repository).

## Content Guidelines
The guidelines for what each directory should contain are as follows.

### The **apps** subdirectories
The following **should** be included in the apps subdirectories:
- README.md : Markdown file that explains everything needed to build, run, and test your application.
- Makefile : [See the Makefile Guidence](MAKEFILE_GUIDANCE.md) for details of how your Makefile should build, prepare, test, and run the application.
- AUTHORS : Text file with names, email addresses, and organizations for all authors of the contribution.
- screen_shot.jpg : An image with pixel width of 200 to represent your application (optional.)

The following should **not** typically be stored in the apps subdirectories:
- Optimized neural networks (OpenVINO IR files - .bin and .xml) : These can be created by invoking the OpenVINO Model Optimizer from the Makefile.
- Neural network models : These should originate from the networks, or caffe, or tensorflow directories.
- Trained neural networks (weights) : Typically these should be downloaded by a Makefile in the neural network's own directory
- Training data : If required, this should be downloaded by the Makefile in one of the subdirectories of the data directory.
- Images files : These should be downloaded to (if many) or stored in (if not too many) a subdirectory of the data directory.

### The **networks**, **caffe** * and **tensorflow** * subdirectories
The following **should** be included in the networks, caffe, and tensorflow subdirectories; use caffe/GoogLeNet as an example to follow:
- README.md : Markdown file that explains how a developer can use the content in the subdirectory.
- Makefile : [See the Makefile Guidence](MAKEFILE_GUIDANCE.md) for details on how neural network Makefiles should work.  In general, they should have targets to: download any large files, optimize neural networks, build, and run a small code example demonstrating how to use the neural network.
- Small example program (optional) : Small C++, or Python program that demonstrates how to use the network with the NCS/NCS 2 device. Only required when existing programs (like apps/simple_classifier_py) aren't sufficient for the neural network.
- Network model definition files (like caffe's .prototxt files): These may be downloaded via the Makefile.
- Network mean information : These may be downloaded via the Makefile.
- AUTHORS : Text file with names, email addresses, and organizations for all authors of the contribution.
 
The following **should not** typically be stored in the networks, caffe * or tensorflow *  subdirectory:
- Optimized neural networks (OpenVINO IR files - .bin and .xml) : These are typically created by invoking the OpenVINO Model Optimizer from the Makefile.
- Trained neural network files : These should be downloaded from their Internet home via the Makefile.
- Training data : If required, this should be downloaded by the Makefile in one of the subdirectories of the data directory.
- Images files : These should be downloaded to (if many) or stored in (if not too many) a subdirectory of the data directory.

*Note that the caffe and tensorflow directories are for legacy compatibility with preivious versions of the ncappzoo.  New neural networks should be placed in the **networks** subdirectory.

## Contribution Licensing
All contributions must be licensed under the MIT license [https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT) unless otherwise stated in a lower-level directory for exceptional cases. The [LICENSE](LICENSE) file in the repository top-level directory provides the MIT license details.

Also, for your contribution to be accepted, each commit must be "Signed-off". This is done by committing using the command `git commit --signoff`.

By signing off your commits, you agree to the following agreement, also known as the [Developer Certificate of Origin](http://developercertificate.org/). It assures everyone that the code you're submitting is yours, or that you have rights to submit it.

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
660 York Street, Suite 102,
San Francisco, CA 94110 USA

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

