# Contributing to the Neural Compute App Zoo (NC App Zoo)

We are glad you want to contribute with a model and/or software for the Intel® Movidius™ NC App Zoo. This is a great way to help the developer community. We encourage submissions from any and all developers interested in the Intel® Movidius™ Neural Compute Stick (Intel® Movidius™ NCS).

## Submitting a Pull Request
To contribute a new app or neural network for the NC App Zoo, fork the repository and add (commit) any of the following new subdirectories containing your content (your content could contain any or all of the following):
- apps/(new app name)
- caffe/(new caffe network name)
- tensorflow/(new tensorflow network name)

Your application code goes under **apps**, any neural network that you have been working with for the Intel Movidius NCS will go under **caffe** or **tensorflow** as appropriate for the framework used.

After you have committed changes to your fork of the App Zoo (always commit using the --signoff option as described in the Contribution Licensing section below), create a pull request for the new directories to be pulled into this repo (the NC App Zoo repository).

## Content Guidelines
The guidelines for what each directory should contain are as follows.

### The **apps** subdirectories
The following **should** be included in the apps subdirectories:
- README.md : Markdown file that explains everything needed to build and run your application.
- Makefile : Builds any source code and downloads or copies (from other repo subdirectories) any required files. If your app needs content from other areas in the repo, like a network graph file, your Makefile should invoke the other Makefile to produce the content needed. You can use the apps/stream_infer/Makefile as the pattern to follow.  Strongly suggested that Makefiles include the following targets as a guideline:
  - make help : Display list targets are available
  - make all : Creates everything needed to run the application including other projects in the repository
  - make run : Runs the application.
  - make clean : Removes all the files in this project directory that may get created when making or running this project.  Should not clean other projects in the repository.
- AUTHORS : Text file with names, email addresses, and organizations for all authors of the contribution.

The following should **not** typically be stored in the apps subdirectories:
- NCS graph files : These can be created from your Makefile.
- Neural network models : These should originate from the caffe or tensorflow directories.
- Trained neural networks (weights) : Typically these should be downloaded by a Makefile in the caffe or tensorflow subdirectories.
- Training data : This should be downloaded by a Makefile a subdirectory of the data directory.
- Images files : These should be downloaded to (if many) or stored in (if not too many) a subdirectory of the data directory.

### The **caffe** subdirectories
The following **should** be included in the caffe subdirectories; use caffe/GoogLeNet as an example to follow:
- README.md : Markdown file that explains how a developer can use the content in the subdirectory.
- Makefile : Should have targets to download any large files, such as the trained network files, compile NCS graph files, build, and run a small code example.
- Small example program (optional) : Small C, C++, or Python program that demonstrates how to use the network with the NCS.
- Network model files (.prototxt) file : It's typically downloaded via the Makefile if you don't own the trained network.
- Network mean information : It's typically downloaded via the Makefile if you don't own the trained network.
- AUTHORS : Text file with names, email addresses, and organizations for all authors of the contribution.

The following should **not** typically be stored in the caffe subdirectory:
- NCS graph files : These should be created via the Makefile, which should invoke the SDK compiler to create the graph file.
- Trained neural network files : These should be downloaded from their Internet home via the Makefile.
- Training data : If needed, training data can be downloaded via the Makefile in a subdirectory of the data directory.
- Images files : These should be downloaded to (if many) or stored in (if not too many) a subdirectory of the data directory. 

### The **tensorflow** subdirectories
The following **should** be included in the tensorflow subdirectories; use tensorflow/inception_v1 as an example to follow:
- README.md : Markdown file that explains how a developer can use the content in the subdirectory.
- Makefile : Should have targets to download any large files, such as the trained network files, compile NCS graph files, build, and run a small code example.
- Small example program (optional) : Small C, C++, or Python program that demonstrates how to use the network with the NCS.
- Network mean information : It's typically downloaded via the Makefile if you don't own the trained network.
- AUTHORS : Text file with names, email addresses, and organizations for all authors of the contribution.

The following should **not** typically be stored in the tensorflow subdirectory:
- NCS graph files : These should be created via the Makefile, which should invoke the SDK compiler to create the graph file.
- Trained neural network files : These should be downloaded from their Internet home via the Makefile.
- Training data : If needed training data can be downloaded via the Makefile in a subdirectory of the data directory.
- Images files : These should be downloaded to (if many) or stored in (if not too many) a subdirectory of the data directory.

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

