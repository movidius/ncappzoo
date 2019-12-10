#!/bin/bash
# Copyright Intel Corporation 2019
# This script will check the current system for required dependencies
# If any are missing, it will prompt to install or provide instructions on installing

echo "Checking system dependencies...\n"
if [[ -f /opt/intel/openvino/bin/setupvars.sh || -f ~/intel/openvino/bin/setupvar.sh ]]; then
    echo "Intel Distribution of OpenVINO is installed."
else
    echo -e "\e[33mThe OpenVINO toolkit might not be installed. If you have built the open-source version of the toolkit and have properly set your environment variables, ignore this message. \
    \nOtherwise, please install the Intel Distribution of OpenVINO toolkit from https://https://software.intel.com/en-us/openvino-toolkit\e[39m"
fi

if [[ -z "${INTEL_OPENVINO_DIR}" ]]; then
    echo "Please source the setupvars.sh script to set the environment variables for the current shell."
    exit 1
fi

# Checks if programs are installed. These programs are also needed by OpenVINO for full functionality, so they should be installed anyways. Otherwise, prompts use to install.
if ! [[ -x "$(command -v python3)" ]]; then 
    echo "Python3 is not installed, please install from your package manager (apt install python3), or ensure your path variables are correct."
    echo "Would you this like script to install python3 from apt? (Supported Distributions only) [y\n]"
    read pythonAnswer
    if [ $pythonAnswer == y ]; then
        echo -e "\e[36msudo apt install python3 python3-dev python3-pip\e[39m"
        sudo apt install python3 python3-dev python3-pip
    else
        echo "Installation skipped. Please install a compatible version of python to continue."
        exit 1;
    fi
elif ! [[ -x "$(command -v pip3)" ]]; then
    echo "pip for Python3 is not installed. Please install from your package manager, or ensure your path variables are correct."
    echo "Would you like this script to install pip3 from apt? (Supported Distributions only) [y\n]"
    read pythonAnswer
    if [ $pythonAnswer == y ]; then
        echo -e "\e[36msudo apt install python3-pip\e[39m"
        sudo apt install python3-pip
    else
        echo "Installation skipped. Please install a compatible version of pip to continue."
        exit 1;
        fi
elif ! [[ -x "$(command -v git)" ]]; then
    echo "git is not installed. Please install from your package manager."
    echo "Would you like to install git from apt? (Supported Distributions only) [y\n]"
    read gitAnswer
    if [ $gitAnswer == y ]; then
        echo -e "\e[36msudo apt install git\e[39m"
        sudo apt install git;
    else
        "Installation skipped. Please install a compatible version of git to continue."
        exit 1;
    fi
elif ! [[ -x "$(command -v g++)" ]]; then
    echo "Necessary compilers may not be installed. Please install the build-essential package from your package manager."
    echo "Would you like this script to install build-essential from apt? (Supported Distributions only) [y\n]"
    read buildAnswer
    if [ $buildAnswer == y ]; then
        echo -e "\e[36msudo apt install build-essential\e[39m"
        sudo apt install build-essential
    else
        "Installation skipped. Please install compatbile C\C++ compilers to continue."
        exit 1;
    fi
else
    echo "Python3, pip3, git, and essential compilers are installed."
fi

#Python Packages: checks for networkx and tensorflow, the two basic packages needed by the Model Optimizer for most of the apps.
if pip3 list --format=columns | grep -i 'networkx\|tensorflow\|numpy'; then
    echo "Minimal Python packages installed"
else   
    echo -e "Python packages not installed. Please install from pip or from the Model Optimzer in OpenVINO, or source the appropriate virutalenv.\n"
    echo "Would you like to install necessary packages? (Requires Intel Distribution of OpenVINO installed in default location) Note: This will install Python packages globally. [y\n]"
    read pipAnswer
    if [ $pipAnswer == 'y' ]; then
        echo -e "\e[36mpip3 install -r /opt/intel/openvino/deployment_tools/model_optimizer/requirements.txt\e[39m"
        pip3 install -r /opt/intel/openvino/deployment_tools/model_optimizer/requirements.txt
    else
        echo -e "Please install necessary packages from pip. \nYou can find a list of the necessary packages in the Model Optimizer directory."
        exit 1
    fi
fi