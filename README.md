# Neural Compute Application Zoo (NC App Zoo)

Welcome to the Intel® Movidius™ Neural Compute App Zoo (NC App Zoo). This repository is a place for any interested developers to share their projects (code and Neural Network content) that make use of the Intel® Movidius™ Neural Compute Stick (Intel® Movidius™ NCS) and associated [Intel® Movidius™ Neural Compute Software Development Kit](http://www.github.com/movidius/ncsdk).
 
The NC App Zoo is a community repository with many content owners and maintainers. All NC App Zoo content is being made available here in a central location for others to download, experiment with, modify, build upon, and learn from.

## NC App Zoo Repository Branches
The projects in this repository depend on the NCSDK and the NCAPI contained within that SDK.  The [NCSDK 2.04 release](https://github.com/movidius/ncsdk/releases/tag/v2.04.00.06) introduced the NCAPI v2 which is not backwards compatible with the orginal NCAPI v1.  This means that applications built around NCAPI v1, including those in the NC App Zoo, will not run without modification when NCAPI v2 is installed on the host machine.

If you are interested in NCSDK2 you can take a look at some of the [changes in NCAPI v2](https://movidius.github.io/ncsdk/ncapi/readme.html) as well as the [NCSDK 2.04 Release Notes](https://movidius.github.io/ncsdk/release_notes.html).

While the transition to NCAPI v2 is going on the NC App Zoo will maintain a **ncsdk2** branch for NCSDK 2.x projects as well as the **master** branch which will continue to contain the NCSDK 1.x projects.
You can use the following git command to use the ncsdk2 branch of the NC App Zoo repo:
```bash
git clone -b ncsdk2 https://github.com/movidius/ncappzoo.git
```
At some point in the future the NCSDK 2.x projects will move to the master as 1.x becomes obsolete.

Also of note is the **[ncappzoo/ncapi2_shim](https://github.com/movidius/ncappzoo/tree/ncsdk2/ncapi2_shim)** project which will allow NCAPI v1 python code to run with NCAPI v2 installed with very little work.  See the README.md file in that directory for more information on how to use the shim.  Many projects in the ncsdk2 branch make use of this shim as well.

## NC App Zoo Repository Layout
The NC App Zoo contains the following top-level directories.  See the README file in each of these directory or just click on the links below to explore the contents of the NC App Zoo.
- **[apps](apps/README.md)** : Applications built to use the Intel Movidius NCS.  **This is a great place to start in the NC App Zoo!**
- **[caffe](caffe/README.md)** : Scripts to download caffe models and compile graphs for use with the NCS
- **[tensorflow](tensorflow/README.md)** : Scripts to download TensorFlow™ models and compile graphs for use with the NCS
- **data** : Data and scripts to download data for use with models and applications that use the NCS

The top-level directories above have subdirectories that hold project content. Each of these project subdirectories has one or more owners that assumes responsibility for it. The [OWNERS](OWNERS) file contains the mapping of subdirectory to owner. 

## Contributing to the Neural Compute Application Zoo
The more contributions to the NC App Zoo, the more successful the community will be! We always encourage everyone with Intel Movidius NCS-related content to share by contributing their applications and model-related work to the NC App Zoo. It's easy to do, and if contributing new content, you will be the owner and maintainer of the content.

See the [CONTRIBUTING.md](CONTRIBUTING.md) file for instructions and guidelines for contributing.

## Licensing
All content in the NC App Zoo is licensed via the [MIT license](https://opensource.org/licenses/MIT) unless specifically stated otherwise in lower-level projects. Individual model and code owners maintain the copyrights for their content, but provide it to the community in accordance with the MIT License.

See the [LICENSE](LICENSE) file in the top-level directory for all licensing details, including reuse and redistribution of content in the NC App Zoo repository.

