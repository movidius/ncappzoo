// Copyright(c) 2017 Intel Corporation. 
// License: MIT See LICENSE file in root directory.


#include <stdio.h>
#include <stdlib.h>
#include <mvnc.h>

int main(int argc, char** argv)
{
    ncStatus_t retCode;
    struct ncDeviceHandle_t* deviceHandlePtr;
    
    // Initialize the device handle
    retCode = ncDeviceCreate(0, &deviceHandlePtr);
    if (retCode != NC_OK)
    {
        // Failed to create the device. Make sure device is plugged into host
        printf("ncDeviceCreate Failed [%d]: Make sure Neural Compute device is plugged into host.\n", retCode);
        exit(-1);
    }

    // Open the device
    retCode = ncDeviceOpen(deviceHandlePtr);
    if (retCode != NC_OK)
    {
        // Failed to open the device
        printf("ncDeviceOpen Failed [%d]: Could not open the device.\n", retCode);
        exit(-1);
    }

    // Device handle is ready to use now, pass it to other API calls as needed...

    
    // deviceHandle is ready to use now.  
    // Pass it to other NC API calls as needed and close it when finished.
    printf("Hello NCS! Device opened normally.\n");

    // Close the device
    retCode = ncDeviceClose(deviceHandlePtr);
    if (retCode != NC_OK)
    {
        // Failed to close the device
        printf("ncDeviceClose Failed [%d]: Could not close the device.\n", retCode);
        exit(-1);
    }

    ncDeviceDestroy(&deviceHandlePtr);
    printf("Goodbye NCS!  Device Closed normally.\n");
    printf("NCS device working.\n");
}
