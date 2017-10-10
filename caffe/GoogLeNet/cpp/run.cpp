// Copyright 2017 Intel Corporation. 
// The source code, information and material ("Material") contained herein is  
// owned by Intel Corporation or its suppliers or licensors, and title to such  
// Material remains with Intel Corporation or its suppliers or licensors.  
// The Material contains proprietary information of Intel or its suppliers and  
// licensors. The Material is protected by worldwide copyright laws and treaty  
// provisions.  
// No part of the Material may be used, copied, reproduced, modified, published,  
// uploaded, posted, transmitted, distributed or disclosed in any way without  
// Intel's prior express written permission. No license under any patent,  
// copyright or other intellectual property rights in the Material is granted to  
// or conferred upon you, either expressly, by implication, inducement, estoppel  
// or otherwise.  
// Any license under such intellectual property rights must be express and  
// approved by Intel in writing. 

#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

#include "fp16.h"
#include <mvnc.h>


// somewhat arbitrary buffer size for the device name
#define NAME_SIZE 100

// graph file name - assume we are running in this directory: ncsdk/examples/caffe/GoogLeNet/cpp
#define GRAPH_FILE_NAME "../graph"

// image file name - assume we are running in this directory: ncsdk/examples/caffe/GoogLeNet/cpp
#define IMAGE_FILE_NAME "../../../data/images/nps_electric_guitar.png"


// 16 bits.  will use this to store half precision floats since C++ has no 
// built in support for it.
typedef unsigned short half;

// GoogleNet image dimensions, network mean values for each channel in BGR order.
const int networkDim = 224;
float networkMean[] = {0.40787054*255.0, 0.45752458*255.0, 0.48109378*255.0};

// Load a graph file
// caller must free the buffer returned.
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


half *LoadImage(const char *path, int reqsize, float *mean)
{
	int width, height, cp, i;
	unsigned char *img, *imgresized;
	float *imgfp32;
	half *imgfp16;

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
	imgfp32 = (float*) malloc(sizeof(*imgfp32) * reqsize * reqsize * 3);
	if(!imgfp32)
	{
		free(imgresized);
		perror("malloc");
		return 0;
	}
	for(i = 0; i < reqsize * reqsize * 3; i++)
		imgfp32[i] = imgresized[i];
	free(imgresized);
	imgfp16 = (half*) malloc(sizeof(*imgfp16) * reqsize * reqsize * 3);
	if(!imgfp16)
	{
		free(imgfp32);
		perror("malloc");
		return 0;
	}
	for(i = 0; i < reqsize*reqsize; i++)
	{
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
	floattofp16((unsigned char *)imgfp16, imgfp32, 3*reqsize*reqsize);
	free(imgfp32);
	return imgfp16;
}


int main(int argc, char** argv)
{
    mvncStatus retCode;
    void *deviceHandle;
    char devName[NAME_SIZE];
    retCode = mvncGetDeviceName(0, devName, NAME_SIZE);
    if (retCode != MVNC_OK)
    {   // failed to get device name, maybe none plugged in.
        printf("No NCS devices found\n");
        exit(-1);
    }
    
    // Try to open the NCS device via the device name
    retCode = mvncOpenDevice(devName, &deviceHandle);
    if (retCode != MVNC_OK)
    {   // failed to open the device.  
        printf("Could not open NCS device\n");
        exit(-1);
    }
    
    // deviceHandle is ready to use now.  
    // Pass it to other NC API calls as needed and close it when finished.
    printf("Successfully opened NCS device!\n");

    // Now read in a graph file
    unsigned int graphFileLen;
    void* graphFileBuf = LoadFile(GRAPH_FILE_NAME, &graphFileLen);

    // allocate the graph
    void* graphHandle;
    retCode = mvncAllocateGraph(deviceHandle, &graphHandle, graphFileBuf, graphFileLen);
    if (retCode != MVNC_OK)
    {   // error allocating graph
        printf("Could not allocate graph for file: %s\n", GRAPH_FILE_NAME); 
        printf("Error from mvncAllocateGraph is: %d\n", retCode);
    }
    else
    {   // successfully allocated graph.  Now graphHandle is ready to go.  
        // use graphHandle for other API calls and call mvncDeallocateGraph
        // when done with it.
        printf("Successfully allocated graph for %s\n", GRAPH_FILE_NAME);

        // LoadImage will read image from disk, convert channels to floats
        // subtract network mean for each value in each channel.  Then, convert 
        // floats to half precision floats and return pointer to the buffer 
        // of half precision floats (Fp16s)
        half* imageBufFp16 = LoadImage(IMAGE_FILE_NAME, networkDim, networkMean);
        
        // calculate the length of the buffer that contains the half precision floats.
        // 3 channels * width * height * sizeof a 16bit float 
        unsigned int lenBufFp16 = 3*networkDim*networkDim*sizeof(*imageBufFp16);

        // start the inference with mvncLoadTensor()
        retCode = mvncLoadTensor(graphHandle, imageBufFp16, lenBufFp16, NULL);
        if (retCode != MVNC_OK)
        {   // error loading tensor
            printf("Could not load tensor\n");
            printf("Error from mvncLoadTensor is: %d\n", retCode);
        }
        else
        {   // the inference has been started, now call mvncGetResult() for the
            // inference result 
            printf("Successfully loaded the tensor for image %s\n", IMAGE_FILE_NAME);
            
            void* resultData16;
            void* userParam;
            unsigned int lenResultData;
            retCode = mvncGetResult(graphHandle, &resultData16, &lenResultData, &userParam);
            if (retCode == MVNC_OK)
            {   // Successfully got the result.  The inference result is in the buffer pointed to by resultData
                printf("Successfully got the inference result for image %s\n", IMAGE_FILE_NAME);
                printf("resultData is %d bytes which is %d 16-bit floats.\n", lenResultData, lenResultData/(int)sizeof(half));
                
                // convert half precision floats to full floats
                int numResults = lenResultData / sizeof(half);
                float* resultData32;
	        resultData32 = (float*)malloc(numResults * sizeof(*resultData32));
                fp16tofloat(resultData32, (unsigned char*)resultData16, numResults);

                float maxResult = 0.0;
                int maxIndex = -1;
                for (int index = 0; index < numResults; index++)
                {
                    // printf("Category %d is: %f\n", index, resultData32[index]);
                    if (resultData32[index] > maxResult)
                    {
                        maxResult = resultData32[index];
                        maxIndex = index;
                    }
                }
                printf("Index of top result is: %d\n", maxIndex);
                printf("Probability of top result is: %f\n", resultData32[maxIndex]);
            } 
        }

        retCode = mvncDeallocateGraph(graphHandle);
	graphHandle = NULL;
    }

    free(graphFileBuf);    
    retCode = mvncCloseDevice(deviceHandle);
    deviceHandle = NULL;
}
