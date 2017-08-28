# mvncCloseDevice()

Type|Function
------------ | -------------
Header|mvnc.h
Library| libmvnc.so
Return|[mvncStatus](mvncStatus.md)
See also|[mvncOpenDevice](mvncOpenDevice.md), [mvncGetDeviceOption](mvncGetDeviceOption.md), [mvncSetDeviceOption](mvncSetDeviceOption.md)

## Overview
This function is used to cease communication and reset the device.

## Prototype

```C
mvncStatus mvncCloseDevice(void *deviceHandle);
```
## Parameters
Name|Type|Description
----|----|-----------
deviceHandle|void*|Pointer to the opaque NCS Device structure that was allocated and returned from the mvncOpenDevice function.

## Return
This function returns an appropriate value from the [mvncStatus](mvncStatus.md) enumeration.

## Gotchas

## Example
```C
TODO
```
