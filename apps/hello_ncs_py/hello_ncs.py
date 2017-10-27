#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory. 

# Python script to open and close a single NCS device

import mvnc.mvncapi as fx

# main entry point for the program
if __name__=="__main__":

     # set the logging level for the NC API
    fx.SetGlobalOption(fx.GlobalOption.LOG_LEVEL, 0)

    # get a list of names for all the devices plugged into the system
    ncs_names = fx.EnumerateDevices()
    if (len(ncs_names) < 1):
        print("Error - no NCS devices detected, verify an NCS device is connected.")
        quit() 


    # get the first NCS device by its name.  For this program we will always open the first NCS device.
    dev = fx.Device(ncs_names[0])

    
    # try to open the device.  this will throw an exception if someone else has it open already
    try:
        dev.OpenDevice()
    except:
        print("Error - Could not open NCS device.")
        quit()


    print("Hello NCS! Device opened normally.")
    

    try:
        dev.CloseDevice()
    except:
        print("Error - could not close NCS device.")
        quit()

    print("Goodbye NCS! Device closed normally.")
    print("NCS device working.")
    
