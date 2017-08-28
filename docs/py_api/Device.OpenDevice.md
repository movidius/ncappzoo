# Device.OpenDevice()

Type|Function
------------ | -------------
Library|mvnc.*
Return|Value 
Revision|0.1
See also|mvnc.Device()<br>mvnc.Device.GetDeviceOption()

## Overview
This function is used to initialize the device.  

## Syntax

```python
devHandle.OpenDevice()
```

## Gotchas

## Example
```python
for devnum in range(len(devices)):
    print("***********************************************")
    devHandle.append(mvnc.Device(devices[devnum]))
    devHandle[devnum].OpenDevice()

    opt = devHandle[devnum].GetDeviceOption(mvnc.DeviceOption.OPTIMISATIONLIST)
    print("Optimisations:")
    print(opt)

    graphHandle.append(devHandle[devnum].AllocateGraph(graph))
    graphHandle[devnum].SetGraphOption(mvnc.GraphOption.ITERATIONS, 1)
    iterations = graphHandle[devnum].GetGraphOption(mvnc.GraphOption.ITERATIONS)
    print("Iterations:", iterations
```
