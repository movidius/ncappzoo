# Copyright 2018 Intel Corporation.
# NPS

import sys
import numpy
from enum import Enum
from mvnc import mvncapi as mvncapi2


class mvncStatus(Enum):
    OK = mvncapi2.Status.OK
    BUSY = mvncapi2.Status.BUSY
    ERROR = mvncapi2.Status.ERROR
    OUT_OF_MEMORY = mvncapi2.Status.OUT_OF_MEMORY
    DEVICE_NOT_FOUND = mvncapi2.Status.DEVICE_NOT_FOUND
    INVALID_PARAMETERS = mvncapi2.Status.DEVICE_NOT_FOUND
    TIMEOUT = mvncapi2.Status.TIMEOUT
    MVCMD_NOT_FOUND = mvncapi2.Status.MVCMD_NOT_FOUND
    NO_DATA = -10008   # depricated
    GONE = -10009      # depricated
    UNSUPPORTED_GRAPH_FILE = mvncapi2.Status.UNSUPPORTED_GRAPH_FILE
    MYRIAD_ERROR = mvncapi2.Status.MYRIAD_ERROR
    NOT_ALLOCATED = mvncapi2.Status.NOT_ALLOCATED
    UNAUTHORIZED = mvncapi2.Status.UNAUTHORIZED
    UNSUPPORTED_CONFIGURATION_FILE = mvncapi2.Status.UNSUPPORTED_CONFIGURATION_FILE
    UNSUPPORTED_FEATURE = mvncapi2.Status.UNSUPPORTED_FEATURE
    INVALID_DATA_LENGTH = mvncapi2.Status.INVALID_DATA_LENGTH
    INVALID_HANDLE = mvncapi2.Status.INVALID_HANDLE


class GlobalOption(Enum):
    LOG_LEVEL = mvncapi2.GlobalOption.RW_LOG_LEVEL
    LOGLEVEL = LOG_LEVEL
    API_VERSION = mvncapi2.GlobalOption.RO_API_VERSION


class DeviceOption(Enum):
    # TEMP_LIM_LOWER = 1                                                         # Not supported
    # TEMP_LIM_HIGHER = 2                                                        # Not supported
    # BACKOFF_TIME_NORMAL = 3                                                    # Not supported
    # BACKOFF_TIME_HIGH = 4                                                      # Not supported
    # BACKOFF_TIME_CRITICAL = 5                                                  # Not supported
    # TEMPERATURE_DEBUG = 6                                                      # Not supported
    THERMAL_STATS = mvncapi2.DeviceOption.RO_THERMAL_STATS                       # numpy.darray of floats,
                                                                                 #   device temps in deg C
    # OPTIMISATION_LIST = 1001                                                   # Not supported
    THERMAL_THROTTLING_LEVEL = mvncapi2.DeviceOption.RO_THERMAL_THROTTLING_LEVEL # Current throttling in play
                                                                                 #   0 no level reached,
                                                                                 #   1 lower guard reached
                                                                                 #   2 upper guard reached
    DEVICE_STATE = mvncapi2.DeviceOption.RO_DEVICE_STATE                         # any of the DeviceState values
    CURRENT_MEMORY_USED = mvncapi2.DeviceOption.RO_CURRENT_MEMORY_USED           # memory in use in bytes
    MEMORY_SIZE = mvncapi2.DeviceOption.RO_MEMORY_SIZE                           # device memory size
    FW_VERSION = mvncapi2.DeviceOption.RO_FW_VERSION                             # numpy.darray 4 unsigned ints
                                                                                 #   [major, minor, hw type, build num]
    DEBUG_INFO = mvncapi2.DeviceOption.RO_DEBUG_INFO                             # Extended debugging info, string.
    MVTENSOR_VERSION = mvncapi2.DeviceOption.RO_MVTENSOR_VERSION                 # numpy.darray uints [major, minor]
    DEVICE_NAME = mvncapi2.DeviceOption.RO_DEVICE_NAME                           # device name, string
    HW_VERSION = mvncapi2.DeviceOption.RO_HW_VERSION                             # returns HW Version, enum


class DeviceState(Enum):
    CREATED = mvncapi2.DeviceState.CREATED
    OPENED = mvncapi2.DeviceState.OPENED
    CLOSED = mvncapi2.DeviceState.CLOSED


class GraphState(Enum):
    CREATED = mvncapi2.GraphState.CREATED
    ALLOCATED = mvncapi2.GraphState.ALLOCATED
    WAITING_FOR_BUFFERS = mvncapi2.GraphState.WAITING_FOR_BUFFERS
    RUNNING = mvncapi2.GraphState.RUNNING


class DeviceHwVersion(Enum):
    MA2450 = mvncapi2.DeviceHwVersion.MA2450
    MA2480 = mvncapi2.DeviceHwVersion.MA2480

class GraphOption(Enum):
    # ITERATIONS = 0                                      # No Longer supported
    # NETWORK_THROTTLE = 1                                # No Longer supported
    # DONT_BLOCK = 2                                      # No Longer supported
    TIME_TAKEN = mvncapi2.GraphOption.RO_TIME_TAKEN       # numpy.darray of floats.  for the last inferece, the times
                                                          # taken at each graph layer
    DEBUG_INFO = mvncapi2.GraphOption.RO_DEBUG_INFO       # extended debug information, string
    GRAPH_NAME = mvncapi2.GraphOption.RO_GRAPH_NAME       # name of the graph, string

    GRAPH_STATE = mvncapi2.GraphOption.RO_GRAPH_STATE     # returns value from GraphState enum
    GRAPH_VERSION = mvncapi2.GraphOption.RO_GRAPH_VERSION # returns numpy.darray of unsigned ints [major, minor]


def EnumerateDevices():
    devices = mvncapi2.enumerate_devices()
    ret_devices = []
    for index in range(len(devices)):
        ret_devices.append(str(index))
    return ret_devices


def SetGlobalOption(opt, data):
    mvncapi2.global_set_option(mvncapi2.GlobalOption(opt.value), data)


def GetGlobalOption(opt):
    return mvncapi2.global_get_option(mvncapi2.GlobalOption(opt.value))

class Device:
    def __init__(self, name):
        self.name = name
        devices = mvncapi2.enumerate_devices()
        index = int(name)
        self._api2_device = mvncapi2.Device(devices[index])

    def OpenDevice(self):
        self._api2_device.open()

    def CloseDevice(self):
        self._api2_device.close()
        self._api2_device.destroy()

    def SetDeviceOption(self, opt, data):
        self._api2_device(mvncapi2.DeviceOption(opt.value), data)

    def GetDeviceOption(self, opt):
        return self._api2_device.get_option(mvncapi2.DeviceOption(opt.value))

    def AllocateGraph(self, graphfile):
        api2_graph = mvncapi2.Graph("mvnc_simple_api graph")
        api2_fifo_in, api2_fifo_out = api2_graph.allocate_with_fifos(self._api2_device, graphfile,
                                                                     input_fifo_data_type=mvncapi2.FifoDataType.FP16,
                                                                     output_fifo_data_type=mvncapi2.FifoDataType.FP16)
        return Graph(api2_graph, api2_fifo_in, api2_fifo_out)

class Graph:
    def __init__(self, api2_graph, api2_fifo_in, api2_fifo_out):
        self._api2_graph = api2_graph
        self._api2_fifo_in = api2_fifo_in
        self._api2_fifo_out = api2_fifo_out
        self.userobjs = {}

    def SetGraphOption(self, opt, value):
        self._api2_graph.set_option(mvncapi2.GraphOption(opt.value), value)

    def GetGraphOption(self, opt):
        return self._api2_graph.get_option(mvncapi2.GraphOption(opt.value))

    def DeallocateGraph(self):
        self._api2_fifo_in.destroy()
        self._api2_fifo_out.destroy()
        self._api2_graph.destroy()

    def LoadTensor(self, tensor, userobj):
        self._api2_graph.queue_inference_with_fifo_elem(self._api2_fifo_in, self._api2_fifo_out, tensor, userobj)

    def GetResult(self):
        output, userobj = self._api2_fifo_out.read_elem()
        return output, userobj
