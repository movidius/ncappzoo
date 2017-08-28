# Introduction
The Movidius™ Neural Compute SDK and Movidius™ Neural Compute Stick (NCS) enable rapid prototyping, validation and deployment of Deep Neural Networks (DNN).

The NCS in used in two primary scenarios:
- Profiling, tuning, and compiling a DNN on a development computer (host system) with the tools provided in the Movidius™ Neural Compute SDK. In this scenario the host system is typically a desktop or laptop machine running Ubuntu 16.04 Desktop (x86, 64 bit) but you can use any supported platform for these steps.

- Prototyping a user application on a development computer (host system) which accesses the hardware of the NCS to accelerate DNN inferences via the API provided with the Movidius™ Neural Compute SDK. In this scenario the host system can be a developer workstation or any developer system that runs an operating system compatible with the API. 

The following diagram shows the typical workflow for development with the NC
![](ncs_workflow.jpg)

The training phase does not utilize the NCS hardware or SDK, while the subsequent phases of “profiling, tuning and compiling” and “prototyping” do require the NCS hardware and the accompanying Movidius™ Neural Compute SDK

The SDK contains a set of software tools to compile, profile, and check validity of your DNN as well as an API for both the C and Python programming languages.  The API is provided to allow users to create software which offloads the neural network computation onto the Movidius™ Neural Compute Stick.

# [Installation and Configuration](install.md)
# [SDK Tools](tools.md)
# [C API](c_api.md)
# [Python API](python_api.md)
# [Trouble Shooting](troubleshooting.md)
# [NCS Forum](forum.md)
. 

.
 
. 

old links below
. 

.
 
. 


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


