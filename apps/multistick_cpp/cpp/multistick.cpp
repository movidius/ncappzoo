// Copyright(c) 2017 Intel Corporation. 
// License: MIT See LICENSE file in root directory. 

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

// from current director to examples base director
// #define APP_BASE_DIR "../"

#define EXAMPLES_BASE_DIR "../../../"

// graph file names - assume the graph file is in the current directory.
#define GOOGLENET_GRAPH_FILE_NAME "googlenet.graph"
#define SQUEEZENET_GRAPH_FILE_NAME "squeezenet.graph"

// image file name - assume we are running in this directory: ncsdk/examples/caffe/GoogLeNet/cpp
#define GOOGLENET_IMAGE_FILE_NAME EXAMPLES_BASE_DIR "data/images/nps_electric_guitar.png"
#define SQUEEZENET_IMAGE_FILE_NAME EXAMPLES_BASE_DIR "data/images/nps_baseball.png"


// 16 bits.  will use this to store half precision floats since C++ has no 
// built in support for it.
typedef unsigned short half;

// GoogleNet image dimensions, network mean values for each channel in BGR order.
const int networkDimGoogleNet = 224;
const int networkDimSqueezeNet = 227;
float networkMeanGoogleNet[] = {0.40787054*255.0, 0.45752458*255.0, 0.48109378*255.0};
float networkMeanSqueezeNet[] = {0.40787054*255.0, 0.45752458*255.0, 0.48109378*255.0};

// Prototypes
void *LoadGraphFile(const char *path, unsigned int *length);
half *LoadImage(const char *path, int reqsize, float *mean);
// end prototypes

// Reads a graph file from the file system and copies it to a buffer
// that is allocated internally via malloc.
// Param path is a pointer to a null terminate string that must be set to the path to the 
//            graph file on disk before calling
// Param length must must point to an integer that will get set to the number of bytes 
//              allocated for the buffer
// Returns pointer to the buffer allcoated. 
// Note: The caller must free the buffer returned.
void *LoadGraphFile(const char *path, unsigned int *length)
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

// Reads an image file from disk (8 bit per channel RGB .jpg or .png or other formats 
// supported by stbi_load.)  Resizes it, subtracts the mean from each channel, and then 
// converts to an array of half precision floats that is suitable to pass to mvncLoadTensor.  
// The returned array will contain 3 floats for each pixel in the image the first float 
// for a pixel is it's the Blue channel value the next is Green and then Red.  The array 
// contains the pixel values in row major order.
// Param path is a pointer to a null terminated string that must be set to the path of the 
//            to read before calling.
// Param reqsize must be set to the width and height that the image will be resized to.  
//               Its assumed width and height are the same size.
// Param mean must be set to point to an array of 3 floating point numbers.  The three
//            numbers are the mean values for the blue, green, and red channels in that order.
//            each B, G, and R value from the image will have this value subtracted from it.
// Returns a pointer to a buffer that is allocated internally via malloc.  this buffer contains
//         the 16 bit float values that can be passed to mvncLoadTensor().  The returned buffer 
//         will contain reqSize*reqSize*3 half floats.
half *LoadImage(const char *path, int reqSize, float *mean)
{
	int width, height, cp, i;
	unsigned char *img, *imgresized;
	float *imgfp32;
	half *imgfp16;

	img = stbi_load(path, &width, &height, &cp, 3);
	if(!img)
	{
		printf("Error - the image file %s could not be loaded\n", path);
		return NULL;
	}
	imgresized = (unsigned char*) malloc(3*reqSize*reqSize);
	if(!imgresized)
	{
		free(img);
		perror("malloc");
		return NULL;
	}
	stbir_resize_uint8(img, width, height, 0, imgresized, reqSize, reqSize, 0, 3);
	free(img);
	imgfp32 = (float*) malloc(sizeof(*imgfp32) * reqSize * reqSize * 3);
	if(!imgfp32)
	{
		free(imgresized);
		perror("malloc");
		return NULL;
	}
	for(i = 0; i < reqSize * reqSize * 3; i++)
		imgfp32[i] = imgresized[i];
	free(imgresized);
	imgfp16 = (half*) malloc(sizeof(*imgfp16) * reqSize * reqSize * 3);
	if(!imgfp16)
	{
		free(imgfp32);
		perror("malloc");
		return NULL;
	}
	for(i = 0; i < reqSize*reqSize; i++)
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
	floattofp16((unsigned char *)imgfp16, imgfp32, 3*reqSize*reqSize);
	free(imgfp32);
	return imgfp16;
}


// Opens one NCS device.
// Param deviceIndex is the zero-based index of the device to open
// Param deviceHandle is the address of a device handle that will be set 
//                    if opening is successful
// Returns true if works or false if doesn't.
bool OpenOneNCS(int deviceIndex, void** deviceHandle)
{
    mvncStatus retCode;
    char devName[NAME_SIZE];
    retCode = mvncGetDeviceName(deviceIndex, devName, NAME_SIZE);
    if (retCode != MVNC_OK)
    {   // failed to get this device's name, maybe none plugged in.
        printf("Error - NCS device at index %d not found\n", deviceIndex);
        return false;
    }
    
    // Try to open the NCS device via the device name
    retCode = mvncOpenDevice(devName, deviceHandle);
    if (retCode != MVNC_OK)
    {   // failed to open the device.  
        printf("Error - Could not open NCS device at index %d\n", deviceIndex);
        return false;
    }
    
    // deviceHandle is ready to use now.  
    // Pass it to other NC API calls as needed and close it when finished.
    printf("Successfully opened NCS device at index %d!\n", deviceIndex);
    return true;
}


// Loads a compiled network graph onto the NCS device.
// Param deviceHandle is the open device handle for the device that will allocate the graph
// Param graphFilename is the name of the compiled network graph file to load on the NCS
// Param graphHandle is the address of the graph handle that will be created internally.
//                   the caller must call mvncDeallocateGraph when done with the handle.
// Returns true if works or false if doesn't.
bool LoadGraphToNCS(void* deviceHandle, const char* graphFilename, void** graphHandle)
{
    mvncStatus retCode;

    // Read in a graph file
    unsigned int graphFileLen;
    void* graphFileBuf = LoadGraphFile(graphFilename, &graphFileLen);

    // allocate the graph
    retCode = mvncAllocateGraph(deviceHandle, graphHandle, graphFileBuf, graphFileLen);
    free(graphFileBuf);
    if (retCode != MVNC_OK)
    {   // error allocating graph
        printf("Could not allocate graph for file: %s\n", graphFilename); 
        printf("Error from mvncAllocateGraph is: %d\n", retCode);
        return false;
    }

    // successfully allocated graph.  Now graphHandle is ready to go.  
    // use graphHandle for other API calls and call mvncDeallocateGraph
    // when done with it.
    printf("Successfully allocated graph for %s\n", graphFilename);

    return true;
}


// Runs an inference and outputs result to console
// Param graphHandle is the graphHandle from mvncAllocateGraph for the network that 
//                   will be used for the inference
// Param imageFileName is the name of the image file that will be used as input for
//                     the neural network for the inference
// Param networkDim is the height and width (assumed to be the same) for images that the
//                     network expects. The image will be resized to this prior to inference.
// Param networkMean is pointer to array of 3 floats that are the mean values for the network
//                   for each color channel, blue, green, and red in that order.
// Returns tru if works or false if doesn't
bool DoInferenceOnImageFile(void* graphHandle, const char* imageFileName, int networkDim, float* networkMean)
{
    mvncStatus retCode;

    // LoadImage will read image from disk, convert channels to floats
    // subtract network mean for each value in each channel.  Then, convert 
    // floats to half precision floats and return pointer to the buffer 
    // of half precision floats (Fp16s)
    half* imageBufFp16 = LoadImage(imageFileName, networkDim, networkMean);

    // calculate the length of the buffer that contains the half precision floats. 
    // 3 channels * width * height * sizeof a 16bit float 
    unsigned int lenBufFp16 = 3*networkDim*networkDim*sizeof(*imageBufFp16);

    // start the inference with mvncLoadTensor()
    retCode = mvncLoadTensor(graphHandle, imageBufFp16, lenBufFp16, NULL);
    free(imageBufFp16);
    if (retCode != MVNC_OK)
    {   // error loading tensor
        printf("Error - Could not load tensor\n");
        printf("    mvncStatus from mvncLoadTensor is: %d\n", retCode);
        return false;
    }

    // the inference has been started, now call mvncGetResult() for the
    // inference result 
    printf("Successfully loaded the tensor for image %s\n", imageFileName);
    
    void* resultData16;
    void* userParam;
    unsigned int lenResultData;
    retCode = mvncGetResult(graphHandle, &resultData16, &lenResultData, &userParam);
    if (retCode != MVNC_OK)
    {
        printf("Error - Could not get result for image %s\n", imageFileName);
        printf("    mvncStatus from mvncGetResult is: %d\n", retCode);
        return false;
    }

    // Successfully got the result.  The inference result is in the buffer pointed to by resultData
    printf("Successfully got the inference result for image %s\n", imageFileName);
    //printf("resultData is %d bytes which is %d 16-bit floats.\n", lenResultData, lenResultData/(int)sizeof(half));

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

// Main entry point for the program
int main(int argc, char** argv)
{
    mvncStatus retCode;
    void* devHandle1;
    void* devHandle2;
    void* graphHandleGoogleNet; 
    void* graphHandleSqueezeNet;

    if (!OpenOneNCS(0, &devHandle1)) 
    {	// couldn't open first NCS device
        exit(-1);
    }
    if (!OpenOneNCS(1, &devHandle2)) 
    {   // couldn't open second NCS device
        exit(-1);
    }

    void* deviceHandle = devHandle1;

    if (!LoadGraphToNCS(devHandle1, GOOGLENET_GRAPH_FILE_NAME, &graphHandleGoogleNet))
    {
        mvncCloseDevice(devHandle1);
        mvncCloseDevice(devHandle2);
        exit(-2);
    }
    if (!LoadGraphToNCS(devHandle2, SQUEEZENET_GRAPH_FILE_NAME, &graphHandleSqueezeNet))
    {
        mvncDeallocateGraph(graphHandleGoogleNet);
        graphHandleGoogleNet = NULL;
        mvncCloseDevice(devHandle1);
        mvncCloseDevice(devHandle2);
        exit(-2);
    }

    printf("\n--- NCS 1 inference ---\n");
    DoInferenceOnImageFile(graphHandleGoogleNet, GOOGLENET_IMAGE_FILE_NAME, networkDimGoogleNet, networkMeanGoogleNet);
    printf("-----------------------\n");

    printf("\n--- NCS 2 inference ---\n");
    DoInferenceOnImageFile(graphHandleSqueezeNet, SQUEEZENET_IMAGE_FILE_NAME, networkDimSqueezeNet, networkMeanSqueezeNet);
    printf("-----------------------\n");

    retCode = mvncDeallocateGraph(graphHandleSqueezeNet);
    graphHandleSqueezeNet = NULL;
    retCode = mvncDeallocateGraph(graphHandleGoogleNet);
    graphHandleGoogleNet = NULL;
  
    retCode = mvncCloseDevice(devHandle1);
    devHandle1 = NULL;

    retCode = mvncCloseDevice(devHandle2);
    devHandle2 = NULL;
}
