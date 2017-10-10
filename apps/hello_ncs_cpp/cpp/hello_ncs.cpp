// Copyright(c) 2017 Intel Corporation. 
// License: MIT See LICENSE file in root directory.


#include <stdio.h>
#include <stdlib.h>

#include <mvnc.h>

// somewhat arbitrary buffer size for the device name
#define NAME_SIZE 100


int main(int argc, char** argv)
{
    mvncStatus retCode;
    void *deviceHandle;
    char devName[NAME_SIZE];
    retCode = mvncGetDeviceName(0, devName, NAME_SIZE);
    if (retCode != MVNC_OK)
    {   // failed to get device name, maybe none plugged in.
        printf("Error - No NCS devices found.\n");
	printf("    mvncStatus value: %d\n", retCode);
        exit(-1);
    }
    
    // Try to open the NCS device via the device name
    retCode = mvncOpenDevice(devName, &deviceHandle);
    if (retCode != MVNC_OK)
    {   // failed to open the device.  
        printf("Error - Could not open NCS device.\n");
	printf("    mvncStatus value: %d\n", retCode);
        exit(-1);
    }
    
    // deviceHandle is ready to use now.  
    // Pass it to other NC API calls as needed and close it when finished.
    printf("Hello NCS! Device opened normally.\n");

    retCode = mvncCloseDevice(deviceHandle);
    deviceHandle = NULL;
    if (retCode != MVNC_OK)
    {
        printf("Error - Could not close NCS device.\n");
	printf("    mvncStatus value: %d\n", retCode);
	exit(-1);
    }

    printf("Goodbye NCS!  Device Closed normally.\n");
    printf("NCS device working.\n");
}
