# Installation and Configuration
This page provides installation and configuration information needed to use the NCS and the examples provided in this repository. To use the NCS you will need to have the Movidius™ Neural Compute SDK installed on your development computer. The SDK installation provides an option to install the examples in this repostitory.  If you've already installed the SDK on your development computer you may have selected the option to also install these examples.  If you have not already installed the SDK you should follow the instructions in the Example Installation with SDK section in this page, and when prompted select the option to install the examples. 

## Prerequisites
To build and run the examples in this repository you will need to have the following.
- Movidius™ Neural Compute Stick (NCS)
- Movidius™ Neural Compute SDK 
- Development computer with Supported OS (currently Ubuntu 16.04 Desktop or Raspian Jessie)
- Internet connection.
- USB Camera (optional)

## Connecting the NCS to a development computer
The NCS connects to the development computer over a USB 2.0 High Speed interface. Plug the NCS directly to a USB port on your development computer or into a powered USB hub that is plugged into your development computer.

![](ncs_plugged.jpg)

## Example Installation with SDK
To install the examples in this repository along with the SDK use the following command on your development computer.  When prompted, select the option to install examples.  If you haven't already installed the SDK on your development computer you should use this command to install.
```
wget http://whereever.com/ncsdk_setup.sh && chmod +x ncsdk_setup.sh && ./ncsdk_setup.sh
```

## Example Installation without SDK 
To install only the examples and not the SDK on you development computer use the following command to clone the repository and then make appropriate examples for your development computer.  If you already have the SDK installed and only need the examples on your machine you should use this command to install. 
```
git clone http://github.com/Movidius/MvNC_Examples && cd MvNC_Examples && make install && make
```

## Installing and Building Individual Examples
Whether installing with the SDK or without it, both methods above will install and build the examples that are appropriate for your development system including prerequisite software.  Each example comes with its own Makefile that will install only that specific example and any prerequisites that it requires.   To install and build any individual example type the following command from within that example's base directory.
```
make install && make 
```
