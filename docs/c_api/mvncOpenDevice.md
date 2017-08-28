# mvncOpenDevice()

Type|Function
------------ | -------------
Header|mvnc.h
Library| libmvnc.so
Return|[mvncStatus](mvncStatus.md)
See also|[mvncCloseDevice](mvncCloseDevice.md), [mvncGetDeviceOption](mvncGetDeviceOption.md), [mvncSetDeviceOption](mvncSetDeviceOption.md)

## Overview
This function is used to initialize the NCS device and return a device handle that can be passed to other API functions.

## Prototype

```C
mvncStatus mvncOpenDevice(const char *name, void **deviceHandle);
```

## Parameters
Name|Type|Description
----|----|------------
name|const char\*|Pointer to a constant array of chars that contains the name of the device to open. This value is obtained from mvncGetDeviceName.
deviceHandle|void \*\*|Address of a pointer that will be set to point to an opaque structure representing an NCS device.

## Return
This function returns an appropriate value from the [mvncStatus](mvncStatus.md) enumeration.

## Gotchas

## Example
```C
TODO
```
