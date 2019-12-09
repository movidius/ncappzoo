#!/bin/bash
# Copyright Intel Corporation 2019
# This script will check the current system for required dependencies
# If any are missing, it will prompt to install or provide instructions on installing

echo "Checking system dependencies...\n"
if [[ -f /opt/intel/openvino/bin/setupvars.sh ]]; then
    echo "Intel Distribution of OpenVINO is installed."
else
    printf "The OpenVINO toolkit might not be installed. If you have built the open-source version of the toolkit and have properly set your environment variables, ignore this message. \
    Otherwise, please install the Intel Distribution of OpenVINO toolkit from https://https://software.intel.com/en-us/openvino-toolkit"
fi

if [[ -z "${INTEL_OPENVINO_DIR}" ]]; then
    echo "Please source the setupvars.sh script to set the environment variables for the current shell."
    exit 1
fi

# Checks if programs are installed. These programs are also needed by OpenVINO for full functionality, so they should be installed anyways. Otherwise, prompts use to install.
if ! [[ -x "$(command -v python3)" ]]; then 
    echo "Python3 is not installed, please install from your package manager (apt install python3), or ensure your path variables are correct."
    echo "Would you like to install python3 from apt? (Supported Distributions only) [y\n]\n"
    read pythonAnswer
    if [ $pythonAnswer == y ]; then
        sudo apt install python3 python3-dev python3-pip
    else
        echo "Installation skipped. Please install a compatible version of python to continue."
        exit 1;
    fi
elif ! [[ -x "$(command -v pip3)" ]]; then
    echo "pip for Python3 is not installed. Please install from your package manager, or ensure your path variables are correct."
    echo "Would you like to install pip3 from apt? (Supported Distributions only) [y\n]"
    read pythonAnswer
    if [ $pythonAnswer == y ]; then
        sudo apt install python3-pip
    else
        echo "Installation skipped. Please install a compatible version of python to continue."
        exit 1;
        fi
elif ! [[ -x "$(command -v git)" ]]; then
    echo "git is not installed. Please install from your package manager."
    echo "Would you like to install pip3 from apt? (Supported Distributions only) [y\n]"
    read gitAnswer
    if [ $gitAnswer == y ]; then
        sudo apt install git;
    else
        "Installation skipped. Please install a compatible version of python to continue."
        exit 1;
    fi
elif ! [[ -x "$(command -v g++)" ]]; then
    echo "Necessary compilers may not be installed. Please install the build-essential package from your package manager."
    echo "Would you like to install build-essential from apt? (Supported Distributions only) [y\n]"
    read buildAnswer
    if [ $buildAnswer == y ]; then
        sudo apt install build-essential
    else
        "Installation skipped. Please install compatbile C\C++ compilers to continue."
        exit 1;
    fi
else
    echo "Python3, pip3, git, and essential compilers are installed."
fi

#Python Packages: checks for networkx and tensorflow, the two basic packages needed by the Model Optimizer for most of the apps.
if pip3 list --format=columns | grep -i 'networkx\|tensorflow'; then
    echo "Minimal Python packages installed"
else   
    echo "Python packages not installed. Please install from pip or from the Model Optimzer in OpenVINO, or source the appropriate virutalenv."
    echo "Would you like to install necessary packages? (Requires Intel Distribution of OpenVINO installed in default location) Note: This will install packages globally. [y\n]"
    read pipanswer
    if [ $pipanswer == 'y' ]; then
        pip3 install -r /opt/intel/openvino/deployment_tools/model_optimizer/requirements.txt
    else
        echo "Please install necessary packages from pip."
        exit 1
    fi
fi