# Movidius Neural Compute SDK C API

The SDK comes with a C Language API that enables applications that utilize hardware accelerated Deep Neural Networks via the Movidius™ Neural Compute Stick (NCS.)  The C API is provided as a header file (mvnc.h) and an associated library file (libmvnc.so) both of which are placed on the development computer when the SDK is installed.  Details of the C API are provided below and within the documents linked from here. 

## Enumerations
### [mvncStatus](mvncStatus.md)
### [mvncDeviceOptions](mvncDeviceOptions.md)
### [mvncGraphOptions](mvncGraphOptions.md)

## Functions
### [mvncGetDeviceName](mvncGetDeviceName.md)
### [mvncOpenDevice](mvncOpenDevice.md)
### [mvncAllocateGraph](mvncAllocateGraph.md)
### [mvncDeallocateGraph](mvncDeallocateGraph.md)
### [mvncLoadTensor](mvncLoadTensor.md)
### [mvncGetResult](mvncGetResult.md)
### [mvncSetGraphOption](mvncSetGraphOption.md)
### [mvncGetGraphOption](mvncGetGraphOption.md)
### [mvncSetDeviceOption](mvncSetDeviceOption.md)
### [mvncGetDeviceOption](mvncGetDeviceOption.md)
### [mvncCloseDevice](mvncCloseDevice.md)

