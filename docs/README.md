# Introduction
The Movidius™ Neural Compute SDK and Movidius™ Neural Compute Stick (NCS) enable rapid prototyping, validation and deployment of Deep Neural Networks (DNN).
The SDK contains parsing program that intelligently converts existing networks, creating an optimal conversion specifically for the Movidius™ architecture.  The SDK also contains a set of C and Python APIs to offload the neural network computation onto the Movidius™ Neural Compute Stick.

* [Architecture Details](https://github.intel.com/pages/MIG-Internal/MvNC_Examples/Architecture/)

# Typical Neural Network Development flow

# Setup
wget http://whereever.com/ncsdk_setup.sh && chmod +x ncsdk_setup.sh && ./ncsdk_setup.sh

OR

git clone http://github.com/Movidius/MvNC_Examples && cd MvNC_Examples && make install

## System Requirements

# ncCompile

# ncProfile

# ncCheck

# C API

# Python API

## Global Methods

* [mvnc.EnumerateDevices](https://github.intel.com/pages/MIG-Internal/MvNC_Examples/API/py_EnumerateDevices/)
* [mvnc.Status](https://github.intel.com/pages/MIG-Internal/MvNC_Examples/API/py_Class_Status/)
* [mvnc.GlobalOption](https://github.intel.com/pages/MIG-Internal/MvNC_Examples/API/py_Class_GlobalOption/)
  * [mvnc.SetGlobalOption()](https://github.intel.com/pages/MIG-Internal/MvNC_Examples/API/py_Class_SetGlobalOption/)
  * [mvnc.GetGlobalOption()](https://github.intel.com/pages/MIG-Internal/MvNC_Examples/API/py_Class_GetGlobalOption/)
* [mvnc.Device](https://github.intel.com/pages/MIG-Internal/MvNC_Examples/API/py_Class_Device/)
  * [mvnc.Device.OpenDevice()](https://github.intel.com/pages/MIG-Internal/MvNC_Examples/API/py_OpenDevice/)
  * [mvnc.Device.CloseDevice()](https://github.intel.com/pages/MIG-Internal/MvNC_Examples/API/py_CloseDevice/)
  * [mvnc.Device.DeviceOption](https://github.intel.com/pages/MIG-Internal/MvNC_Examples/API/py_Class_DeviceOption/)
    * [mvnc.Device.DeviceOption.SetDeviceOption()](https://github.intel.com/pages/MIG-Internal/MvNC_Examples/API/py_SetDeviceOption/)
    * [mvnc.Device.DeviceOption.GetDeviceOption()](https://github.intel.com/pages/MIG-Internal/MvNC_Examples/API/py_GetDeviceOption/)
  * [mvnc.Device.Graph](https://github.intel.com/pages/MIG-Internal/MvNC_Examples/API/py_Class_Graph/)
    * [mvnc.Device.AllocateGraph()](https://github.intel.com/pages/MIG-Internal/MvNC_Examples/API/py_AllocateGraph/)  
    * [mvnc.Device.DeallocateGraph()](https://github.intel.com/pages/MIG-Internal/MvNC_Examples/API/py_DeallocateGraph/)  
    * [mvnc.Device.LoadTensor()](https://github.intel.com/pages/MIG-Internal/MvNC_Examples/API/py_LoadTensor/)  
    * [mvnc.Device.GetResult()](https://github.intel.com/pages/MIG-Internal/MvNC_Examples/API/py_GetResult/)  
    * [mvnc.GraphOption](https://github.intel.com/pages/MIG-Internal/MvNC_Examples/API/py_Class_GraphOption/)
      * [mvnc.Device.SetGraphOption()](https://github.intel.com/pages/MIG-Internal/MvNC_Examples/API/py_SetGraphOption/)  
      * [mvnc.Device.GetGraphOption()](https://github.intel.com/pages/MIG-Internal/MvNC_Examples/API/py_GetGraphOption/)  

# Examples
