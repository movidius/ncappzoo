// Copyright(c) 2017 Intel Corporation. 
// License: MIT See LICENSE file in root directory.

#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

#include <mvnc.h>


// graph file name - assume we are running in this directory: ncsdk/examples/caffe/GoogLeNet/cpp
#define GRAPH_FILE_NAME "../graph"

// image file name - assume we are running in this directory: ncsdk/examples/caffe/GoogLeNet/cpp
#define IMAGE_FILE_NAME "../../../data/images/nps_electric_guitar.png"


// GoogleNet image dimensions, network mean values for each channel in BGR order.
const int networkDim = 224;
float networkMean[] = {0.40787054*255.0, 0.45752458*255.0, 0.48109378*255.0};

// Load a graph file
// caller must free the buffer returned.
// path is the full or relative path to the graph file
// length is the number of bytes read upon return
void *LoadFile(const char *path, unsigned int *length)
{
    FILE *fp;
    char *buf;

    fp = fopen(path, "rb");
    if(fp == NULL)
	    return 0;
    fseek(fp, 0, SEEK_END);
    *length = ftell(fp);
    rewind(fp);
    if(!(buf = (char*) malloc(*length)))
    {
	    fclose(fp);
	    return 0;
    }
    if(fread(buf, 1, *length, fp) != *length)
    {
	    fclose(fp);
	    free(buf);
	    return 0;
    }
    fclose(fp);
    return buf;
}


// Assumption that a float is a 32 bit value
// read an image file from the file system and put it into 
// and convert it to an array of floats
// path is a null terminated full or relative path to the image file
//      The file is assumed to be a .jpg or .png
// reqsize is the dimension the image should be resized to.  It is  
//         assumed to be square so this is the height and width dimension
// mean is an array of 3 floats that are the network mean for each channel.
//      this is in BGR order.
// bufSize If not NULL will be set to the number of bytes allocated which will be:
//         sizeof(float) * reqsize * reqsize * 3;
float *LoadImage32(const char *path, int reqsize, float *mean, unsigned int* bufSize)
{
    if (bufSize != NULL) 
    {
        *bufSize = 0;
    }
    int width, height, cp, i;
    unsigned char *img, *imgresized;
    float *imgfp32;

    img = stbi_load(path, &width, &height, &cp, 3);
    if(!img)
    {
        printf("The picture %s could not be loaded\n", path);
        return 0;
    }
    imgresized = (unsigned char*) malloc(3*reqsize*reqsize);
    if(!imgresized)
    {
        free(img);
        perror("malloc");
        return 0;
    }
    stbir_resize_uint8(img, width, height, 0, imgresized, reqsize, reqsize, 0, 3);
    free(img);
    unsigned int allocateSize = sizeof(*imgfp32) * reqsize * reqsize * 3;
    if (bufSize != NULL)
    {
        *bufSize = allocateSize;
    }
    imgfp32 = (float*) malloc(allocateSize);
    if(!imgfp32)
    {
        if (bufSize != NULL)
        {
            *bufSize = 0;
        }
        free(imgresized);
        perror("malloc");
        return 0;
    }
    for(i = 0; i < reqsize * reqsize * 3; i++)
    {
	    imgfp32[i] = imgresized[i];
    }
    free(imgresized);
    for(i = 0; i < reqsize*reqsize; i++)
    {
        // imgfp32 comes in RGB order but network expects to be in
        // BRG order so convert to BGR here while subtracting the mean.
        float blue, green, red;
        blue = imgfp32[3*i+2];
        green = imgfp32[3*i+1];
        red = imgfp32[3*i+0];

        imgfp32[3*i+0] = blue-mean[0];
        imgfp32[3*i+1] = green-mean[1]; 
        imgfp32[3*i+2] = red-mean[2];

        // uncomment to see what values are getting passed to mvncLoadTensor() before conversion to half float
        //printf("Blue: %f, Grean: %f,  Red: %f \n", imgfp32[3*i+0], imgfp32[3*i+1], imgfp32[3*i+2]);
    }
    return imgfp32;
}


// The entry point for the program
int main(int argc, char** argv)
{
    ncStatus_t retCode;
    struct ncDeviceHandle_t* deviceHandlePtr;


    // Try to create the first Neural Compute device (at index zero) 
    retCode = ncDeviceCreate(0, &deviceHandlePtr);
    if (retCode != NC_OK)
    {   // failed to create the device.  
        printf("Could not create NC device\n");
        exit(-1);
    }

    // deviceHandle is created and ready to be opened
    retCode = ncDeviceOpen(deviceHandlePtr);
    if (retCode != NC_OK)
    {   // failed to open the device.  
        printf("Could not open NC device\n");
        exit(-1);
    }

    // The device is open and ready to be used.
    // Pass it to other NC API calls as needed and close and destroy it when finished.
    printf("Successfully opened NC device!\n");

    // Create the graph
    struct ncGraphHandle_t* graphHandlePtr;
    retCode = ncGraphCreate("GoogLeNet Graph", &graphHandlePtr);
    if (retCode != NC_OK)
    {   // error allocating graph
        printf("Could not create graph.\n"); 
        printf("Error from ncGraphCreate is: %d\n", retCode);
    }
    else
    {   // successfully created graph.  Now we need to destory it when finished with it.
        // Now we need to allocate graph and create and in/out fifos
        struct ncFifoHandle_t* inFifoHandlePtr = NULL;
        struct ncFifoHandle_t* outFifoHandlePtr = NULL;

        // Now read in a graph file from disk to memory buffer and then allocate the graph based on the file we read
        unsigned int graphFileLen;
        void* graphFileBuf = LoadFile(GRAPH_FILE_NAME, &graphFileLen);
        retCode = ncGraphAllocateWithFifos(deviceHandlePtr, graphHandlePtr, graphFileBuf, graphFileLen, &inFifoHandlePtr, &outFifoHandlePtr);
        free(graphFileBuf);

        if (retCode != NC_OK)
        {   // error allocating graph or fifos
            printf("Could not allocate graph with fifos.\n");  
            printf("Error from ncGraphAllocateWithFifos is: %d\n", retCode);
        }
        else
        {
            // Now graphHandle is ready to go we it can now process inferences.  
            printf("Successfully allocated graph for %s\n", GRAPH_FILE_NAME);            

            // assumption here that floats are single percision 32 bit.
            unsigned int tensorSize = 0;  /* size of image buffer should be: sizeof(float) * reqsize * reqsize * 3;*/
            float* imageBufFP32Ptr = LoadImage32(IMAGE_FILE_NAME, networkDim, networkMean, NULL);
            tensorSize = sizeof(float) * networkDim * networkDim * 3;

            // queue the inference to start, when its done the result will be placed on the output fifo 
            retCode = ncGraphQueueInferenceWithFifoElem(
                          graphHandlePtr, inFifoHandlePtr, outFifoHandlePtr, imageBufFP32Ptr, &tensorSize, NULL);
            if (retCode != NC_OK)
            {   // error queuing input tensor for inference
                printf("Could not queue inference\n");
                printf("Error from ncGraphQueueInferenceWithFifoElem is: %d\n", retCode);
            }
            else
            {   // the inference has been started, now read the output queue for the
                // inference result 
                printf("Successfully queued the inference for image %s\n", IMAGE_FILE_NAME);

                // get the size required for the output tensor.  This depends on the 
                // network definition as well as the output fifo's data type.  
                // if the network outputs 1000 tensor elements and the fifo 
                // is using FP32 (float) as the data type then we need a buffer of 
                // sizeof(float) * 1000 into which we can read the inference results.
                // 
                // Rather than calculate this size we can also query the fifo itself
                // for this size with the fifo option NC_RO_FIFO_ELEMENT_DATA_SIZE.
                unsigned int outFifoElemSize = 0; 
                unsigned int optionSize = sizeof(outFifoElemSize);
                ncFifoGetOption(outFifoHandlePtr,  NC_RO_FIFO_ELEMENT_DATA_SIZE,
                                &outFifoElemSize, &optionSize);

                float* resultDataFP32Ptr = (float*) malloc(outFifoElemSize); 
                void* UserParamPtr = NULL;

                // read the output of the inference.  this will be in FP32 since that is how the 
                // fifos are created by default.
                retCode = ncFifoReadElem(outFifoHandlePtr, (void*)resultDataFP32Ptr, &outFifoElemSize, &UserParamPtr);
                if (retCode == NC_OK)
                {   // Successfully got the inference result.  
                    // The inference result is in the buffer pointed to by resultDataFP32Ptr
                    printf("Successfully got the inference result for image %s\n", IMAGE_FILE_NAME);
                    int numResults = outFifoElemSize/(int)sizeof(float);

                    printf("resultData is %d bytes which is %d 32-bit floats.\n", outFifoElemSize, numResults);

                    float maxResult = 0.0;
                    int maxIndex = -1;
                    for (int index = 0; index < numResults; index++)
                    {
                        if (resultDataFP32Ptr[index] > maxResult)
                        {
                            maxResult = resultDataFP32Ptr[index];
                            maxIndex = index;
                        }
                    }
                    printf("Index of top result is: %d\n", maxIndex);
                    printf("Probability of top result is: %f\n", resultDataFP32Ptr[maxIndex]);
                } 
                free((void*)resultDataFP32Ptr);
            }
        }
        ncFifoDestroy(&inFifoHandlePtr);
        ncFifoDestroy(&outFifoHandlePtr);
        ncGraphDestroy(&graphHandlePtr);
        ncDeviceClose(deviceHandlePtr);
        ncDeviceDestroy(&deviceHandlePtr);
    }
}





