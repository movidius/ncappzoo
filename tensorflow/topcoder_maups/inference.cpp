#include <bits/stdc++.h>
//#include <stdio.h>
//#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "fp16.h"
#include <mvnc.h>

using namespace std;
using namespace cv;

// somewhat arbitrary buffer size for the device name
#define NAME_SIZE 100

// graph file names - assume the graph file is in the current directory.
#define GRAPH_FILE_NAME "compiled.graph"

// 16 bits.  will use this to store half precision floats since C++ has no 
// built in support for it.
typedef unsigned short half;

// image dimensions, network mean values for each channel in BGR order.
const int networkDim = 299;

void *LoadGraphFile(const char *path, unsigned int *length) {
	FILE *fp;
	char *buf;

	fp = fopen(path, "rb");
	if(fp == NULL)
		return 0;
	fseek(fp, 0, SEEK_END);
	*length = ftell(fp);
	rewind(fp);
	if(!(buf = (char *) malloc(*length))) {
		fclose(fp);
		return 0;
	}
	if(fread(buf, 1, *length, fp) != *length) {
		fclose(fp);
		free(buf);
		return 0;
	}
	fclose(fp);

	return buf;
}

half *LoadImage(const char *path, int reqSize) {
	Mat img = imread(path, IMREAD_COLOR);
	Rect roi = Rect((img.cols<img.rows?0:(img.cols-img.rows)/2), (img.rows<img.cols?0:(img.rows-img.cols)/2), min(img.cols,img.rows), min(img.cols,img.rows));
	Mat crop = img(roi), r;
	resize(crop, r, Size(reqSize,reqSize));

	float *imgfp32 = (float *) malloc(sizeof(*imgfp32)*reqSize*reqSize*3);
	half *imgfp16 = (half *) malloc(sizeof(*imgfp16)*reqSize*reqSize*3);

	for(int y=0; y < reqSize; y++)
		for(int x=0; x < reqSize; x++)
			for(int c=0; c < 3; c++)
				imgfp32[(y*reqSize+x)*3+c] = r.at<Vec3b>(y,x)[c]/127.5-1.0;

	floattofp16((unsigned char *) imgfp16, imgfp32, 3*reqSize*reqSize);
	free(imgfp32);

	return imgfp16;
}

bool OpenOneNCS(int deviceIndex, void **deviceHandle) {
	mvncStatus retCode;
	char devName[NAME_SIZE];
	retCode = mvncGetDeviceName(deviceIndex, devName, NAME_SIZE);
	if(retCode != MVNC_OK) {
		printf("Error - NCS device at index %d not found\n", deviceIndex);
		return false;
	}

	retCode = mvncOpenDevice(devName, deviceHandle);
	if(retCode != MVNC_OK) {
		printf("Error - Could not open NCS device at index %d\n", deviceIndex);
		return false;
	}

	return true;
}

bool LoadGraphToNCS(void *deviceHandle, const char *graphFilename, void **graphHandle) {
	mvncStatus retCode;

	unsigned int graphFileLen;
	void *graphFileBuf = LoadGraphFile(graphFilename, &graphFileLen);

	retCode = mvncAllocateGraph(deviceHandle, graphHandle, graphFileBuf, graphFileLen);
	free(graphFileBuf);
	if(retCode != MVNC_OK) {
		printf("Could not allocate graph for file: %s\n", graphFilename); 
		printf("Error from mvncAllocateGraph is: %d\n", retCode);
		return false;
	}

	return true;
}

bool DoInferenceOnImageFile(void *graphHandle, const char *imageFileName, int networkDim) {
	mvncStatus retCode;
	half *imageBufFp16 = LoadImage(imageFileName, networkDim);

	unsigned int lenBufFp16 = 3*networkDim*networkDim*sizeof(*imageBufFp16);

	retCode = mvncLoadTensor(graphHandle, imageBufFp16, lenBufFp16, NULL);
	free(imageBufFp16);
	if(retCode != MVNC_OK) {
		printf("Error - Could not load tensor\n");
		printf("    mvncStatus from mvncLoadTensor is: %d\n", retCode);
		return false;
	}

	void *resultData16;
	void *userParam;
	unsigned int lenResultData;
	retCode = mvncGetResult(graphHandle, &resultData16, &lenResultData, &userParam);
	if(retCode != MVNC_OK) {
		printf("Error - Could not get result for image %s\n", imageFileName);
		printf("    mvncStatus from mvncGetResult is: %d\n", retCode);
		return false;
	}

	int numResults = lenResultData/sizeof(half);
	float *resultData32;
	resultData32 = (float *) malloc(numResults*sizeof(*resultData32));
	fp16tofloat(resultData32, (unsigned char *) resultData16, numResults);

	float maxResult[5] = {-0.1,-0.1,-0.1,-0.1,-0.1};
	int maxIndex[5] = {-1,-1,-1,-1,-1};
	for(int i=0; i < numResults; i++)
		for(int j=0; j < 5; j++) {
			if(resultData32[i] > maxResult[j]) {
				for(int k=4; k > j; k--) {
					maxResult[k] = maxResult[k-1];
					maxIndex[k] = maxIndex[k-1];
				}
				maxResult[j] = resultData32[i];
				maxIndex[j] = i;
				break;
			}
		}
	for(int i=0; i < 5; i++)
		printf("%d,%f,", maxIndex[i]+1, maxResult[i]);

	float *time;
	unsigned int sizeOfValue;
	retCode = mvncGetGraphOption(graphHandle, MVNC_TIME_TAKEN, (void*)&time, &sizeOfValue);
	if(retCode == MVNC_OK) {
		sizeOfValue /= sizeof(float);
		double sum=0.0;
		for(int i=0; i < sizeOfValue; i++)
			sum += time[i];
		printf("%lf\n", sum);
	}
}

int main(int argc, char** argv) {
	mvncStatus retCode;
	void *devHandle1;
	void *graphHandle; 

	if(!OpenOneNCS(0, &devHandle1)) {
		exit(-1);
	}

	if(!LoadGraphToNCS(devHandle1, GRAPH_FILE_NAME, &graphHandle)) {
		mvncCloseDevice(devHandle1);
		exit(-2);
	}

	for(int i=1; i <= 2000; i++) {
		string filename = "provisional_", num = to_string(i);
		while(num.size() < 5)
			num = "0"+num;
		filename += num;
		filename += ".jpg";
		string full = "./provisional/"+filename;
		cout << filename << ",";
		DoInferenceOnImageFile(graphHandle, full.c_str(), networkDim);
	}

	retCode = mvncDeallocateGraph(graphHandle);
	graphHandle = NULL;
	retCode = mvncCloseDevice(devHandle1);
	devHandle1 = NULL;
}
