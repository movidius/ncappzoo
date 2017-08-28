# EnumerateDevices()

Type|Function
------------ | -------------
Library|mvnc.*
Return|Value
Revision|0.1
See also|mvnc.Device()<br>mvnc.Device.GetDeviceOption()

## Overview
This function is used to get a list of the names of the devices present in the system.

## Syntax

```python
devHandle.EnumerateDevices()
```

## Gotchas

## Example
```Python
deviceNames = mvnc.EnumerateDevices()
if len(deviceNames) == 0:
	print("Error - No devices detected.")
```
