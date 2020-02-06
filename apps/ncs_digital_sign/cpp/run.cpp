/*
 * NCS_Digital_Sign
 *
 * Contributing Authors: Christian Canales <christian.canales@intel.com>, Tome Vang <tome.vang@intel.com>, Neal Smith <neal.p.smith@intel.com>, Heather McCabe <heather.m.mccabe@intel.com>, Andrew Herrold <andrew.herrold@intel.com>
 *
 *
 *
 */

#include <iostream>
#include <vector>
#include <time.h>
#include <iomanip>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <inference_engine.hpp>

#define WINDOW_NAME "Ncappzoo NCS Digital Sign - OpenVINO"
#define CAM_SOURCE 0

// window height and width 4:3 ratio
#define WINDOW_WIDTH 500
#define WINDOW_HEIGHT 500


using namespace cv;
using namespace InferenceEngine;

// Location of age and gender networks
const std::string FACE_XML_PATH = "../face-detection-retail-0004.xml";
const std::string FACE_BIN_PATH = "../face-detection-retail-0004.bin";
const std::string AGEGEN_XML_PATH = "../age-gender-recognition-retail-0013.xml";
const std::string AGEGEN_BIN_PATH = "../age-gender-recognition-retail-0013.bin";

bool flag = true;
bool full = false; 
int genCount = 0;
int maleGlobalCount = 0;
int femaleGlobalCount = 0;
int age = 0;
std::vector<int> ageVector;

// text colors and font
const int FONT = cv::FONT_HERSHEY_PLAIN;
const int FONT_SIZE = 2;
const cv::Scalar BLUE = cv::Scalar(255, 0, 0, 255);
const cv::Scalar GREEN = cv::Scalar(0, 255, 0, 255);
const cv::Scalar PINK = Scalar(255, 80, 180, 255);
const float FACE_DET_THRESHOLD = 0.65;
const int MAX_FACES_DETECTED = 10;
const float GENDER_CONF_THRESHOLD = 60.0;

// time to wait between face detection/inferences
const double INFERENCE_INTERVAL = 0.03;
const unsigned int FEMALE_LABEL = 0;
const unsigned int MALE_LABEL = 1;

struct detectionResults{
	Mat croppedMat;
	float xmin;
	float ymin;
	float xmax;
	float ymax;
	int age = 18;
	int ageConf = 0.0;
	std::string gender;
	float genderConf = 0.0;

};



int calculateAgeAvg(std::vector<int> ageVector){
    int num = ageVector.size();
    //std::cout << "NUM = " << num << std::endl;
    int avg = ageVector[0];
    //std::cout << "START = " << avg << std::endl;
    std::string::size_type size = ageVector.size();
    for (unsigned i = 1; i < size; i++){
        avg = avg + ageVector[i];
    }
    int finalAvg = avg/num;
    return finalAvg;
}

void showImg(int age, int gender) {
	cv::namedWindow("Advertisement", 0);
	cv::setWindowProperty("Advertisement", 0, 1);
	if (genCount > 50){
	if (gender == 1) {
		if (age < 17) {
			cv::Mat image = cv::imread("../advertisement_images/AD_young_boy.jpg");
			cv::Mat resizeImg;
			cv::resize(image, resizeImg, cv::Size(1280, 720));
			cv::imshow("Advertisement", resizeImg);
			cv::waitKey(1);
		}
		else if (age > 17 && age < 35) {
			cv::Mat image = cv::imread("../advertisement_images/AD_male.jpg");
			cv::Mat resizeImg;
			cv::resize(image, resizeImg, cv::Size(1280, 720));
			cv::imshow("Advertisement", resizeImg);
			cv::waitKey(1);
		}
		else if (age > 35 && age < 50) {
			cv::Mat image = cv::imread("../advertisement_images/AD_male.jpg");
			cv::Mat resizeImg;
			cv::resize(image, resizeImg, cv::Size(1280, 720));
			cv::imshow("Advertisement", resizeImg);
			cv::waitKey(1);
		}
		else {
			cv::Mat image = cv::imread("../advertisement_images/AD_male.jpg");
			cv::Mat resizeImg;
			cv::resize(image, resizeImg, cv::Size(1280, 720));
			cv::imshow("Advertisement", resizeImg);
			cv::waitKey(1);
		}
	}
	else {
		if (age < 17) {
			cv::Mat image = cv::imread("../advertisement_images/AD_young_girl.jpg");
			cv::Mat resizeImg;
			cv::resize(image, resizeImg, cv::Size(1280, 720));
			cv::imshow("Advertisement", resizeImg);
			cv::waitKey(1);
		}
		else if (age > 17 && age < 35) {
			cv::Mat image = cv::imread("../advertisement_images/AD_woman.jpg");
			cv::Mat resizeImg;
			cv::resize(image, resizeImg, cv::Size(1280, 720));
			cv::imshow("Advertisement", resizeImg);
			cv::waitKey(1);
		}
		else if (age > 35 && age < 50) {
			cv::Mat image = cv::imread("../advertisement_images/AD_woman.jpg");
			cv::Mat resizeImg;
			cv::resize(image, resizeImg, cv::Size(1280, 720));
			cv::imshow("Advertisement", resizeImg);
			cv::waitKey(1);
		}
		else {
			cv::Mat image = cv::imread("../advertisement_images/AD_woman.jpg");
			cv::Mat resizeImg;
			cv::resize(image, resizeImg, cv::Size(1280, 720));
			cv::imshow("Advertisement", resizeImg);
			cv::waitKey(1);
		}
	}
	genCount = 0;
    }
}



void getTopResults(unsigned int numberOfResultsToReturn, InferenceEngine::Blob& input, std::vector<unsigned> &output) {
    auto scores = input.buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
 
    
    if (numberOfResultsToReturn > input.size())
    {
        std::cout << "The number of desired results is greater than total number of results." << '\n';
        std::cout << "Setting number of desired results equal to total number of results." << '\n';
        numberOfResultsToReturn = input.size();
    }
    else if (numberOfResultsToReturn <= 0)
    {
        std::cout << "The number of desired results is less than or equal to zero." << '\n';
        std::cout << "Setting number of desired results to 1." << '\n';
    }
    // Create a vector of indexes
    std::vector<unsigned> classIndexes(input.size());
    std::iota(std::begin(classIndexes), std::end(classIndexes), 0);
    std::partial_sort(std::begin(classIndexes), std::end(classIndexes), std::end(classIndexes), 
            [&scores](unsigned left, unsigned right){
                return scores[left] > scores[right];});
    output.resize(numberOfResultsToReturn);
    for (unsigned int j = 0; j < numberOfResultsToReturn; j++) 
    {
        output.at(j) = classIndexes.at(j);
    }
       
}


void calculateAdvertisement(int maleTotal, int femaleTotal, int avgAge){
    int total = maleTotal + femaleTotal;
    double malePercent = (double)maleTotal/(double)total;
    std::cout << "MALE PERCENT: " << malePercent << std::endl;
    double femalePercent = (double)femaleTotal/(double)total;
	std::cout << "FEMALE PERCENT: " << femalePercent << std::endl;

    if (malePercent > 0.70){
        double mpercent = (double)malePercent * 100.00;
        std::cout << "Male Percentage : " << mpercent << "%" << std::endl << "MALE ADVERTISEMENT" << std::endl;
        showImg(avgAge,1);
    } else if (femalePercent > 0.70){
        double fpercent = (double)femalePercent * 100.00;
        std::cout << "Female Percentage : " << fpercent << "%" << std::endl << "FEMALE ADVERTISEMENT" << std::endl;
        showImg(avgAge,0);
    } else {
        flag = true;
    }
}

/*
 * Start.
 */
int main (int argc, char** argv) {
    //
    VideoCapture capture;
    Mat imgIn;
    std::vector <detectionResults> detectedFaces; // vector used to hold results
	std::vector <cv::Scalar> resultColor;
	std::vector <std::string> resultText;
	
    int key;

    // Times for inference timer
    clock_t start_time, elapsed_time;
    
    // Set up the camera
    capture.open(CAM_SOURCE);
    
	const int width  = (int) capture.get(cv::CAP_PROP_FRAME_WIDTH);
	const int height = (int) capture.get(cv::CAP_PROP_FRAME_HEIGHT);

    // Set up the display window
    namedWindow(WINDOW_NAME, WINDOW_NORMAL);
    resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT);
    setWindowProperty(WINDOW_NAME, WND_PROP_ASPECT_RATIO, WINDOW_KEEPRATIO);
    moveWindow(WINDOW_NAME, 0, 0);


    // ---------------------Load MKLDNN Plugin for Inference Engine-----------------------------------------
    Core ie;
    CNNNetwork faceNetwork = ie.ReadNetwork(FACE_XML_PATH, FACE_BIN_PATH);
    CNNNetwork ageGenNetwork = ie.ReadNetwork(AGEGEN_XML_PATH, AGEGEN_BIN_PATH);
    
    // Check network input for face detection
    InputsDataMap faceInputDataMap(faceNetwork.getInputsInfo());
	OutputsDataMap faceOutputDataMap(faceNetwork.getOutputsInfo());
	if (faceInputDataMap.size() != 1 && faceOutputDataMap.size() != 1)
    	throw std::logic_error("Sample supports clean SSD network with one input and one output");

	// Check network input for age and gender
	InputsDataMap ageGenInputDataMap(ageGenNetwork.getInputsInfo());
	OutputsDataMap ageGenOutputDataMap(ageGenNetwork.getOutputsInfo());
	if (ageGenInputDataMap.size() != 1 && ageGenOutputDataMap.size() != 2)
    	throw std::logic_error("Sample supports age gender network with one input and two outputs");

    // -----------------------------Prepare input blobs-----------------------------------------------------
	InputInfo::Ptr& faceInputInfo = faceInputDataMap.begin()->second;
    faceInputInfo->setPrecision(Precision::U8);
    std::string faceInputLayerName = faceInputDataMap.begin()->first;
	faceInputInfo->setPrecision(Precision::U8);
	
	InputInfo::Ptr& ageGenInputInfo = ageGenInputDataMap.begin()->second;
    ageGenInputInfo->setPrecision(Precision::U8);
    std::string ageGenInputLayerName = ageGenInputDataMap.begin()->first;
	ageGenInputInfo->setPrecision(Precision::U8);
	
    // -----------------------------Prepare output blobs-----------------------------------------------------

	auto faceOutputInfo = faceOutputDataMap.begin()->second;
	std::string faceOutputLayerName = faceOutputDataMap.begin()->first;
	faceOutputInfo->setPrecision(Precision::FP32);

	// age gender network output setup
    auto it = ageGenOutputDataMap.begin();
	DataPtr ptrAgeOutput = (it++)->second;
	DataPtr ptrGenderOutput = (it++)->second;
	std::string ageOutputLayerName = ptrAgeOutput->getName();
	std::string genOutputLayerName = ptrGenderOutput->getName();
	for (auto& output : ageGenOutputDataMap) {
    	output.second->setPrecision(Precision::FP32);
	}

    // -------------------------Loading age/gender networks to the plugin----------------------------------
    
    // create executable network object for inference
    std::map<std::string, std::string> config = {{ PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES }};
    auto executableFaceNetwork = ie.LoadNetwork(faceNetwork, "MYRIAD");
    auto executableAgeGenNetwork = ie.LoadNetwork(ageGenNetwork, "MYRIAD");
    
    // create inference requests
    auto faceInferRequest = executableFaceNetwork.CreateInferRequestPtr();
    auto ageGenInferRequest = executableAgeGenNetwork.CreateInferRequestPtr();

    // set the input blobs
    auto faceInput = faceInferRequest->GetBlob(faceInputLayerName);
    auto ageGenInput = ageGenInferRequest->GetBlob(ageGenInputLayerName);
    // 
    auto faceInputData = faceInput->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();
    auto ageGenInputData = ageGenInput->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();


    unsigned int frame_count = 0;

    // skip a frame after this many frames. adjust this if getting laggy camera 
    const int SKIP_AFTER = 3;
    printf("\nStarting ncs_digital_sign app...\n");
    printf("\nPress any key to quit.\n");
    
    // -------------------------Running the inferences----------------------------------
        // Get the current time; inferences will only be performed periodically
    start_time = clock();
    
    // main loop
    while (true) 
    {
        // Get a frame from the camera
        capture >> imgIn;
        // Skip if the frame count equals or exceeds the SKIP_AFTER value
    	if (frame_count++ >= SKIP_AFTER) {
    	    capture >> imgIn;
                frame_count = 0;
        }

        // Flip the image horizontally
        flip(imgIn, imgIn, 1);

		// check if time to do an inference
        elapsed_time = clock() - start_time;
        if ((double)elapsed_time/(double)CLOCKS_PER_SEC >= INFERENCE_INTERVAL) 
        {
        	resultColor.clear();
		    resultText.clear();
        	detectedFaces.clear();
		    
		    // get pointers to the input and output dimensions for the face detection network
			auto faceInputDims = faceInferRequest->GetBlob(faceInputLayerName)->getTensorDesc().getDims();
			auto faceOutputDims = faceInferRequest->GetBlob(faceOutputLayerName)->getTensorDesc().getDims();

			// face network input dimensions
		    unsigned int faceChannelsNumber = faceInputDims.at(1);
		    unsigned int faceInputHeight = faceInputDims.at(2);
		    unsigned int faceInputWidth = faceInputDims.at(3);
		    // face network output dimensions
		    unsigned int maxProposalCount = faceOutputDims.at(2);
		    unsigned int objectSize = faceOutputDims.at(3);
		    
		    Mat imgInput;
		    cv::resize(imgIn, imgInput, cv::Size(faceInputHeight, faceInputWidth));

		    // get the input dimensions for each network
		    size_t faceImageSize = faceInputHeight * faceInputWidth;
		    // set the input data for the age network
		    for (size_t pid = 0; pid < faceImageSize; ++pid) {
		        for (size_t ch = 0; ch < faceChannelsNumber; ++ch) {
		            faceInputData[ch * faceImageSize + pid] = imgInput.at<cv::Vec3b>(pid)[ch];
		        }
		    }

		    
		    // Running the request synchronously 
		    faceInferRequest->Infer();
		    
		    // ------------- Face detection network -----------------
		    auto faceOutput = faceInferRequest->GetBlob(faceOutputLayerName);

		    const float *detections = faceOutput->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
		    for (unsigned int i = 0; i < maxProposalCount; i++) 
		    {
		        float image_id = detections[i * objectSize + 0];
		        if (image_id < 0) 
		        {
		            //std::cout << "Only " << i << " proposals found" << std::endl;
		            break;
		        }

		        float confidence = detections[i * objectSize + 2];

		        int xmin = (int)(detections[i * objectSize + 3] * width);
		        int ymin = (int)(detections[i * objectSize + 4] * height);
		        int xmax = (int)(detections[i * objectSize + 5] * width);
		        int ymax = (int)(detections[i * objectSize + 6] * height);

				// filter out low scores
		        if (confidence > FACE_DET_THRESHOLD) {

		            xmin = std::max(0, xmin);
		            ymin = std::max(0, ymin);
		            xmax = std::min(width, xmax);
		            ymax = std::min(height, ymax);
		            cv::Mat croppedImage = imgIn(Rect(cv::Point2f(xmin, ymin), cv::Point2f(xmax, ymax)));
		            // helper for current detection
		            detectionResults currentDetection;
		            // save the cropped image to send to age/gender
		            currentDetection.croppedMat = croppedImage;
		            // save the current face location 
		            currentDetection.xmin = xmin;
		            currentDetection.ymin = ymin;
		            currentDetection.xmax = xmax;
		            currentDetection.ymax = ymax;
		            // put image into the detected faces vector
		            detectedFaces.push_back(currentDetection);

		        }
		    }
        
		    // ------------- Age Gender network -----------------
			
		    int numInferAgeGen = detectedFaces.size();
		    
			auto ageGenInputDims = ageGenInferRequest->GetBlob(ageGenInputLayerName)->getTensorDesc().getDims();
		    
		    // age gender input dims
		    unsigned int ageGenChannelsNumber = ageGenInputDims.at(1);
		    unsigned int ageGenInputHeight = ageGenInputDims.at(2);
		    unsigned int ageGenInputWidth = ageGenInputDims.at(3);

		    size_t ageGenImageSize = ageGenInputHeight * ageGenInputWidth;
		    
		    
		    for (int i = 0; i < numInferAgeGen; i++)
		    {
		    	cv::resize(detectedFaces.at(i).croppedMat, detectedFaces.at(i).croppedMat, cv::Size(ageGenInputHeight, ageGenInputWidth));
			// set the input data for the gender network
		        for (size_t pid = 0; pid < ageGenImageSize; ++pid) 
		        {
		            for (size_t ch = 0; ch < ageGenChannelsNumber; ++ch) 
		            {
		                ageGenInputData[ch * ageGenImageSize + pid] = detectedFaces.at(i).croppedMat.at<cv::Vec3b>(pid)[ch];
		            }
		        }
		        
		        ageGenInferRequest->Infer();                
				auto ageOutput = ageGenInferRequest->GetBlob(ageOutputLayerName);
				auto ageOutputData = ageOutput->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
			
				auto genOutput = ageGenInferRequest->GetBlob(genOutputLayerName);
				auto genOutputData = genOutput->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();

				unsigned int results_to_display = 1;
				std::vector<unsigned> ageResults;
				std::vector<unsigned> genResults;

				getTopResults(results_to_display, *ageOutput, ageResults);
				getTopResults(results_to_display, *genOutput, genResults);

				auto ageConf = ageOutputData[ageResults[0]]*100;
				auto genConf = genOutputData[genResults[0]]*100;		

				if (genConf > GENDER_CONF_THRESHOLD && genResults.at(0) == FEMALE_LABEL)
				{
					std::string gender = "f";
					detectedFaces.at(i).gender = "Female";
					resultColor.push_back(PINK);
					femaleGlobalCount++;
				}
				else if (genConf > GENDER_CONF_THRESHOLD && genResults.at(0) == MALE_LABEL)
				{
					std::string gender = "m";
					detectedFaces.at(i).gender = "Male";
					resultColor.push_back(BLUE);
					maleGlobalCount++;
				}
				else 
				{
					std::string gender = "x";
					detectedFaces.at(i).gender = "Unknown";
					resultColor.push_back(GREEN);
				}
				resultText.push_back("Age: " + std::to_string((int)(ageConf)));
				std::string age1 = std::to_string((int)(ageConf));
				int age = std::stoi(age1);
				ageVector.push_back(age);
				elapsed_time = clock() - start_time;

			}
			start_time = clock();	
			int total2 =  femaleGlobalCount + maleGlobalCount; 
			if (total2 == 20){
        			int ageAvg = calculateAgeAvg(ageVector);
        			std::cout << "AGE: " << ageAvg << std::endl;
        			calculateAdvertisement(maleGlobalCount, femaleGlobalCount, ageAvg);
        			maleGlobalCount = 0;
        			femaleGlobalCount = 0;
   			}			
			
        }
        
        for (unsigned int i = 0; i < detectedFaces.size(); i++)
        {
        	cv::putText(imgIn, resultText.at(i), cv::Point2f(detectedFaces.at(i).xmin, detectedFaces.at(i).ymin) , FONT, FONT_SIZE, resultColor.at(i), 2);

			cv::rectangle(imgIn, cv::Point2f(detectedFaces.at(i).xmin, detectedFaces.at(i).ymin), cv::Point2f(detectedFaces.at(i).xmax, detectedFaces.at(i).ymax), resultColor.at(i), 1);
        }


	if (true) {
		if (genCount == 100){
			cv::namedWindow("Advertisement", 0);
			cv::moveWindow("General Advertisement", 20, 20);
			cv::setWindowProperty("Advertisement", 0, 1);
			cv::Mat ad = cv::imread("../advertisement_images/AD_gender_neutral.jpg");
			cv::imshow("Advertisement", ad);
		}
		if (flag == true) {
			cv::namedWindow("Advertisement", 0);
			cv::moveWindow("General Advertisement", 20, 20);
			cv::setWindowProperty("Advertisement", 0, 1);
			cv::Mat ad = cv::imread("../advertisement_images/AD_gender_neutral.jpg");
			cv::imshow("Advertisement", ad);
			flag = false;
		}

		cv::namedWindow("Live Feed");
		cv::moveWindow("Live Feed", 0, 0);
		cv::imshow("Live Feed", imgIn);
	}
	else {
		if (flag == true) {
			cv::namedWindow("Advertisement", 0);
			//cv::moveWindow("General Advertisement", 20, 20);
			cv::setWindowProperty("Advertisement", 0, 1);
			cv::Mat ad = cv::imread("../advertisement_images/AD_gender_neutral.jpg");
			cv::imshow("Advertisement", ad);
			flag = false;
		}
		if (genCount == 100){
			cv::namedWindow("Advertisement", 0);
			cv::moveWindow("General Advertisement", 20, 20);
			cv::setWindowProperty("Advertisement", 0, 1);
			cv::Mat ad = cv::imread("../advertisement_images/AD_gender_neutral.jpg");
			cv::imshow("Advertisement", ad);
		}
	}
	genCount = genCount + 1;
        

        // Show the image in the window
        //imshow(WINDOW_NAME, imgIn);
        
        // If the user presses the break key exit the loop
        key = waitKey(1);
        if (key != -1) {
            break;
        }

    } // end main while loop

    // Close all windows
    destroyAllWindows();
    std::cout << "Finished." << std::endl;

    return 0;
}
