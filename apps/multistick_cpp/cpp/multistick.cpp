// Copyright(c) 2017-2018 Intel Corporation. 
// License: MIT See LICENSE file in root directory. 

#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

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
#define SQUEEZENET_IMAGE_FILE_NAME EXAMPLES_BASE_DIR "data/images/nps_electric_guitar.png"
//#define SQUEEZENET_IMAGE_FILE_NAME EXAMPLES_BASE_DIR "data/images/nps_baseball.png"


// GoogleNet image dimensions, network mean values for each channel in BGR order.
const int networkDimGoogleNet = 224;
const int networkDimSqueezeNet = 227;
float networkMeanGoogleNet[] = {0.40787054*255.0, 0.45752458*255.0, 0.48109378*255.0};
float networkMeanSqueezeNet[] = {0.40787054*255.0, 0.45752458*255.0, 0.48109378*255.0};

// Prototypes
void *LoadGraphFile(const char *path, unsigned int *length);
float *LoadImage32(const char *path, int reqsize, float *mean, unsigned int* imageBufSize);
bool OpenOneNCS(int deviceIndex, struct ncDeviceHandle_t** addrOfdeviceHandlePtr);
bool LoadGraphToNCS(struct ncDeviceHandle_t* deviceHandlePtr, const char* graphFilename, struct ncGraphHandle_t** addrOfGraphHandlePtr,
                    struct ncFifoHandle_t** addrOfInFifoHandlePtr, struct ncFifoHandle_t** addrOfOutFifoHandlePtr);
bool DoInferenceOnImageFile(struct ncGraphHandle_t* graphHandlePtr, struct ncFifoHandle_t* inFifoHandlePtr, 
                            struct ncFifoHandle_t* outFifoHandlePtr, 
                            const char* imageFileName, int networkDim, float* networkMean);
// end prototypes

// Reads a graph file from the file system and copies it to a buffer
// that is allocated internally via malloc.
// Param path is a pointer to a null terminated string that must be set to the path to the 
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
// converts to an array of floats that is suitable to pass to ncFifoWriteElem or 
// ncGraphQueueInferenceWithFifoElem.  
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
// Param imageBufSize If not NULL will be set to the number of bytes allocated which will be:
//         sizeof(float) * reqsize * reqsize * 3;
// Returns a pointer to a buffer that is allocated internally via malloc.  This buffer contains
//         the float values that can be passed to the graph for inferencing.  The returned buffer 
//         will contain reqSize*reqSize*3 floats.
float *LoadImage32(const char *path, int reqsize, float *mean, unsigned int* imageBufSize)
{
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
        return NULL;
    }
    stbir_resize_uint8(img, width, height, 0, imgresized, reqsize, reqsize, 0, 3);
    free(img);
    unsigned int mallocSize = sizeof(*imgfp32) * reqsize * reqsize * 3;
    if (imageBufSize != NULL)
    {
        *imageBufSize = mallocSize;
    }
    imgfp32 = (float*) malloc(mallocSize);
    if(!imgfp32)
    {
        free(imgresized);
        perror("malloc");
        return NULL;
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

        // uncomment to see what values will be passed to the graph for inference
        //printf("Blue: %f, Grean: %f,  Red: %f \n", imgfp32[3*i+0], imgfp32[3*i+1], imgfp32[3*i+2]);
    }
    return imgfp32;
}


// Opens one NCS device.
// Param deviceIndex is the zero-based index of the device to open
// Param deviceHandle is the address of a device handle that will be set 
//                    if opening is successful
// Returns true if works or false if doesn't.
bool OpenOneNCS(int deviceIndex, struct ncDeviceHandle_t** addrOfDeviceHandlePtr)
{
    ncStatus_t retCode;

    // Initialize the device handle
    retCode = ncDeviceCreate(deviceIndex, addrOfDeviceHandlePtr);
    if (retCode != NC_OK)
    {
        // Failed to create the device... maybe it isn't plugged in to the host
        printf("Error - ncDeviceCreate failed for for device at index %d error %d\n", deviceIndex, retCode);
        exit(-1);
    }

    // Open the device
    retCode = ncDeviceOpen(*addrOfDeviceHandlePtr);
    if (retCode != NC_OK)
    {
        // Failed to open the device
        printf("Error - ncDeviceOpen failed could not open the device at index %d, error: %d.\n", deviceIndex, retCode);
        ncDeviceDestroy(addrOfDeviceHandlePtr);
        exit(-1);
    }

    // deviceHandle is ready to use now.  
    // Pass it to other NC API calls as needed and close it when finished.
    printf("Successfully opened NCS device at index %d!\n", deviceIndex);

    return true;
}


// Loads a compiled network graph onto the NCS device and create FIFOs to read and write inference 
// input and output.
// Param deviceHandlePtr is the open device handle for the device that will allocate the graph
// Param graphFilename is the name of the compiled network graph file to load on the NCS
// Param addrOfGraphHandlePtr is the address of the graph handle that will be created internally.
//                            the caller must call destroy it when done with the handle.
// Param addrOfInFifoHandlePtr is the address of a FIFO handle that will be created internally
//                             to write inference input to.  The caller must destroy it when 
//                             done with the handle
// Param addrOfOutFifoHandlePtr is the address of a FIFO handle that will be created internally
//                              to read inference results from.  The caller must destroy it when 
//                              done with the handle
// Returns true if works or false if doesn't.
bool LoadGraphToNCS(struct ncDeviceHandle_t* deviceHandlePtr, const char* graphFilename, struct ncGraphHandle_t** addrOfGraphHandlePtr,
                    struct ncFifoHandle_t** addrOfInFifoHandlePtr, struct ncFifoHandle_t** addrOfOutFifoHandlePtr)
{
    ncStatus_t retCode;

    // Create graph 
    retCode = ncGraphCreate("My Graph", addrOfGraphHandlePtr);
    if (retCode != NC_OK)
    {
        printf("Could not create graph for file: %s\n", graphFilename); 
        printf("Error from ncGraphCreate is: %d\n", retCode);
        return false;
    }

    // allocate graph with fifos 
    unsigned int graphSizeInBytes = 0;
    void* graphInMemoryPtr = LoadGraphFile(graphFilename, &graphSizeInBytes);
    retCode = ncGraphAllocateWithFifos(deviceHandlePtr, *addrOfGraphHandlePtr, graphInMemoryPtr, graphSizeInBytes, addrOfInFifoHandlePtr, addrOfOutFifoHandlePtr);
    free(graphInMemoryPtr);
    if (retCode != NC_OK)
    {   // Could not allocate graph 
        printf("Could not allocate graph with fifos: %s\n", graphFilename); 
        printf("Error from ncGraphAllocateWithFifos is: %d\n", retCode);
        ncGraphDestroy(addrOfGraphHandlePtr);
    }

    // successfully allocated graph and FIFOs.  Now graph handle and FIFOs are ready to go.  
    printf("Successfully allocated graph and FIFOs for %s\n", graphFilename);

    return true;
}


// Runs an inference and outputs result to console
// Param graphHandlePtr is pointer to the ncGraphHandle_t that has been created and allocated 
//                      will be used to execute the inference
// Param inFifoHandlePtr is a pointer to a FIFO handle created for input to the graph
// Param outFifoHandlePtr is a pointer to a FIFO handle created for output from the graph
// Param imageFileName is the name of the image file that will be used as input for
//                     the neural network for the inference
// Param networkDim is the height and width (assumed to be the same) for images that the
//                     network expects. The image will be resized to this prior to inference.
// Param networkMean is pointer to array of 3 floats that are the mean values for the network
//                   for each color channel, blue, green, and red in that order.
// Returns tru if works or false if doesn't
bool DoInferenceOnImageFile(struct ncGraphHandle_t* graphHandlePtr, struct ncFifoHandle_t* inFifoHandlePtr, 
                            struct ncFifoHandle_t* outFifoHandlePtr, 
                            const char* imageFileName, int networkDim, float* networkMean)
{
    ncStatus_t retCode;

    void* imageInMemoryPtr = NULL;
    unsigned int imageBufSize = 0;
    imageInMemoryPtr = LoadImage32(imageFileName, networkDim, networkMean, &imageBufSize);

    // queue the inference to start, when its done the result will be placed on the output fifo             
    retCode = ncGraphQueueInferenceWithFifoElem(graphHandlePtr, inFifoHandlePtr, outFifoHandlePtr, imageInMemoryPtr, &imageBufSize, NULL);
    free(imageInMemoryPtr);
    imageInMemoryPtr = NULL;
    if (retCode != NC_OK)
    {   // Could not queue inference
        printf("Error - Could not queue Inference With Fifo Element.  Error: %d\n", retCode);
        return false;
    }

    // inference is queued, now we need to read output FIFO for result

    printf("Successfully queued inference with FIFO elements OK!\n");

    void* outputPtr = NULL;
    unsigned int fifoOutputSize = 0;
    unsigned int optionDataLen = sizeof(unsigned int);
    ncFifoGetOption(outFifoHandlePtr, NC_RO_FIFO_ELEMENT_DATA_SIZE, &fifoOutputSize, &optionDataLen);

    outputPtr = malloc(fifoOutputSize);
    retCode = ncFifoReadElem(outFifoHandlePtr, outputPtr, &fifoOutputSize, NULL);
    if (retCode != NC_OK)
    {
        printf("Error - Could not read Inference result from Fifo Element.  Error: %d\n", retCode);
        free(outputPtr);
        return false;
    }

    // find top classification result.
    int numResults = fifoOutputSize / sizeof(float);
    float* resultData32 = (float*)outputPtr;
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
    free(outputPtr);
    outputPtr = NULL;
}


// Main entry point for the program
int main(int argc, char** argv)
{
    // two devices
    ncDeviceHandle_t* devHandle1Ptr;
    ncDeviceHandle_t* devHandle2Ptr;

    // googlenet graph and in/out FIFOs
    ncGraphHandle_t* graphHandleGoogleNetPtr; 
    ncFifoHandle_t* inFifoHandleGoogleNetPtr;
    ncFifoHandle_t* outFifoHandleGoogleNetPtr;

    // squeezenet graph and in/out FIFOs
    ncGraphHandle_t* graphHandleSqueezeNetPtr;
    ncFifoHandle_t* inFifoHandleSqueezeNetPtr;
    ncFifoHandle_t* outFifoHandleSqueezeNetPtr;


    // open first device to run GoogLeNet
    if (!OpenOneNCS(0, &devHandle1Ptr)) 
    {	// couldn't open first NCS device
        exit(-1);
    }

    // open second device to run SqueezeNet
    if (!OpenOneNCS(1, &devHandle2Ptr)) 
    {   // couldn't open second NCS device
        exit(-1);
    }


    // Load googlenet graph on device 1, this creates graph and 2 fifo handles
    if (!LoadGraphToNCS(devHandle1Ptr, GOOGLENET_GRAPH_FILE_NAME, &graphHandleGoogleNetPtr, 
                        &inFifoHandleGoogleNetPtr, &outFifoHandleGoogleNetPtr))
    {   // error with graph, so clean up and exit
        ncDeviceClose(devHandle1Ptr);
        ncDeviceDestroy(&devHandle1Ptr);
        ncDeviceClose(devHandle2Ptr);
        ncDeviceDestroy(&devHandle2Ptr);
        exit(-2);
    }

    // load squeezenet graph on device 2, this creates a graph and 2 fifo handles
    if (!LoadGraphToNCS(devHandle2Ptr, SQUEEZENET_GRAPH_FILE_NAME, &graphHandleSqueezeNetPtr, 
                        &inFifoHandleSqueezeNetPtr, &outFifoHandleSqueezeNetPtr))
    {   // error with graph, clean up and exit
        ncFifoDestroy(&inFifoHandleGoogleNetPtr);
        ncFifoDestroy(&outFifoHandleGoogleNetPtr);
        ncGraphDestroy(&graphHandleGoogleNetPtr);

        ncDeviceClose(devHandle1Ptr);
        ncDeviceDestroy(&devHandle1Ptr);

        ncDeviceClose(devHandle2Ptr);
        ncDeviceDestroy(&devHandle2Ptr);

        exit(-2);
    }


    printf("\n--- NCS 1 inference ---\n");
    DoInferenceOnImageFile(graphHandleGoogleNetPtr, inFifoHandleGoogleNetPtr, outFifoHandleGoogleNetPtr, GOOGLENET_IMAGE_FILE_NAME, networkDimGoogleNet, networkMeanGoogleNet);
    printf("-----------------------\n");

    printf("\n--- NCS 2 inference ---\n");
    DoInferenceOnImageFile(graphHandleSqueezeNetPtr, inFifoHandleSqueezeNetPtr, outFifoHandleSqueezeNetPtr, SQUEEZENET_IMAGE_FILE_NAME, networkDimSqueezeNet, networkMeanSqueezeNet);
    printf("-----------------------\n");


    // everything worked, clean up all devices, graphs, fifos and we are done.
    ncFifoDestroy(&inFifoHandleGoogleNetPtr);
    ncFifoDestroy(&outFifoHandleGoogleNetPtr);
    ncGraphDestroy(&graphHandleGoogleNetPtr);
    ncDeviceClose(devHandle1Ptr);
    ncDeviceDestroy(&devHandle1Ptr);

    ncFifoDestroy(&inFifoHandleSqueezeNetPtr);
    ncFifoDestroy(&outFifoHandleSqueezeNetPtr);
    ncGraphDestroy(&graphHandleSqueezeNetPtr);
    ncDeviceClose(devHandle2Ptr);
    ncDeviceDestroy(&devHandle2Ptr);
}
