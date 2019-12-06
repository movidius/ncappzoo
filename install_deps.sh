#!/bin/bash
# Copyright Intel Corporation 2019
# This script will check the current system for required dependencies
# If any are missing, it will prompt to install or provide instructions on installing

echo "Checking system dependencies...\n"
if [[ -f /opt/intel/openvino/bin/setupvars.sh ]]; then
    echo "Intel Distribution of OpenVINO is installed."
else
    echo "The OpenVINO toolkit might not be installed. If you have built the open-source version of the toolkit and have properly set your environment variables, ignore this message. \
    \n Otherwise, please install the Intel Distribution of OpenVINO toolkit from https://https://software.intel.com/en-us/openvino-toolkit"
fi

if [[ -z "${INTEL_OPENVINO_DIR}" ]]; then
    echo "Please source the setupvars.sh script to set the environment variables for the current shell."
    exit 1
fi

if ! [[ -x "$(command -v python3)" ]]; then 
    echo "Python3 is not installed, please install from your package manager (apt install python3), or ensure your path variables are correct."
    echo "Would you like to install python3 from apt? (Supported Distributions only) [y\n]\n"
    read pythonAnswer
    if [ pythonAnswer == y ]; then
        sudo apt install python3 python3-dev python3-pip
    else
        echo "Installation skipped. Please install a compatible version of python to continue."
        exit 1;
    fi
elif ! [[ -x "$(command -v pip3)" ]]; then
    echo "pip for Python3 is not installed. Please install from your package manager, or ensure your path variables are correct."
    echo "Would you like to install pip3 from apt? (Supported Distributions only) [y\n]"
    read pythonAnswer
    if [ pythonAnswer == y ;] then
        sudo apt install python3-pip
    else
        echo "Installation skipped. Please install a compatible version of python to continue."
        exit 1;
        fi
elif ! [[ -x "$(command -v git)" ]]; then
    echo "git is not installed. Please install from your package manager."
    echo "Would you like to install pip3 from apt? (Supported Distributions only) [y\n]"
    read gitAnswer
    if [ gitAnswer == y ]; then
        sudo apt install git;
    else
        "Installation skipped. Please install a compatible version of python to continue."
        exit 1;
    fi
elif ! [[ -x "$(command -v g++)" ]]; then
    echo "Necessary compilers may not be installed. Please install the build-essential package from your package manager."
    echo "Would you like to install build-essential from apt? (Supported Distributions only) [y\n]"
    read buildAnswer
    if [ buildAnswer == y ]; then
        sudo apt install build-essential
    else
        "Installation skipped. Please install compatbile C\C++ compilers to continue."
        exit 1;
    fi
fi


#todo - fix this
if [[ "pip3 list --format=columns | grep 'networkx\|tensorflow'" ]]; then
    echo "Python packages installed"
else   
    echo "python packages not installed"
fi