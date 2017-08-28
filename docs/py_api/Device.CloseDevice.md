# Device.CloseDevice()

Type|Function
------------ | -------------
Library|mvnc.*
Return|Value
Revision|0.1
See also|mvnc.Device()<br>mvnc.Device.GetDeviceOption()

## Overview
This function is used to cease communication and reset the device.

## Syntax

```python
devHandle.CloseDevice()
```

## Gotchas

## Example
```Python
def fnc:
	device = mvnc.Device(device_id)
	device.OpenDevice()
	opt = devHandle[devnum].GetDeviceOption(mvnc.DeviceOption.OPTIMISATIONLIST)
	graph = device.AllocateGraph()
	### Do stuff with graph file
	graph.DeallocateGraph()
	device.CloseDevice()
```
