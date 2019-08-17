/*
 * Gender_Age
 *
 * Contributing Authors: Tome Vang <tome.vang@intel.com>, Neal Smith <neal.p.smith@intel.com>, Heather McCabe <heather.m.mccabe@intel.com>
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

#define WINDOW_NAME "Ncappzoo Gender Age - OpenVINO"
#define CAM_SOURCE 0


// Location of age and gender networks
#define FACE_NETWORK_PATH "../face-detection-retail-0004.xml"
#define AGEGEN_NETWORK_PATH "../age-gender-recognition-retail-0013.xml"

// window height and width 4:3 ratio
#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 960


using namespace cv;
using namespace InferenceEngine;

// text colors and font
const int FONT = cv::FONT_HERSHEY_PLAIN;
const int FONT_SIZE = 2;
const cv::Scalar BLUE = cv::Scalar(255, 0, 0, 255);
const cv::Scalar GREEN = cv::Scalar(0, 255, 0, 255);
const cv::Scalar PINK = Scalar(255, 80, 180, 255);

// detection thresholds and constants
const float FACE_DET_THRESHOLD = 0.65;
const int MAX_FACES_DETECTED = 10;
const float GENDER_CONF_THRESHOLD = 60.0;

const unsigned int FEMALE_LABEL = 0;
const unsigned int MALE_LABEL = 1;

// time to wait between face detection/inferences
const double INFERENCE_INTERVAL = 0.03;


// detection struct to hold results
struct detectionResults{
    Mat croppedMat;  // cropped face mat
    float xmin;      // coordinate of bounding box
    float ymin;      // coordinate of bounding box
    float xmax;      // coordinate of bounding box
    float ymax;      // coordinate of bounding box
    int ageConf = 0; // age confidence aka age of the person
    float genderConf = 0.0; // gender confidence

};


/*
 * read a network
 */
InferenceEngine::CNNNetwork readNetwork(String inputNetworkPath) {
    CNNNetReader network_reader;
    network_reader.ReadNetwork(inputNetworkPath);
    network_reader.ReadWeights(inputNetworkPath.substr(0, inputNetworkPath.size() - 4) + ".bin");
    network_reader.getNetwork().setBatchSize(1);
    CNNNetwork network = network_reader.getNetwork();
    return network;
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
    capture.set(CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH);
    capture.set(CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT);
    
    const int width  = (int) capture.get(cv::CAP_PROP_FRAME_WIDTH);
    const int height = (int) capture.get(cv::CAP_PROP_FRAME_HEIGHT);

    // Set up the display window
    namedWindow(WINDOW_NAME, WINDOW_NORMAL);
    resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT);
    setWindowProperty(WINDOW_NAME, WND_PROP_ASPECT_RATIO, WINDOW_KEEPRATIO);
    moveWindow(WINDOW_NAME, 0, 0);

	// Create the inference engine object from the inference engine core api
    Core ie;

    // -------------------------Read network and check network inputs-----------------------------------------------------------
    // Declare the networks
    CNNNetwork faceNetwork;
    CNNNetwork ageGenNetwork;

    // Read the network from the xml file
    faceNetwork = readNetwork(FACE_NETWORK_PATH);
    ageGenNetwork = readNetwork(AGEGEN_NETWORK_PATH);
    
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
    // Get the face network input information, set the precision, get the face network input node name
    InputInfo::Ptr& faceInputInfo = faceInputDataMap.begin()->second;
    faceInputInfo->setPrecision(Precision::U8);
    std::string faceInputLayerName = faceInputDataMap.begin()->first;
    
    // Get the face network input information, set the precision, get the age-gender network input node name
    InputInfo::Ptr& ageGenInputInfo = ageGenInputDataMap.begin()->second;
    ageGenInputInfo->setPrecision(Precision::U8);
    std::string ageGenInputLayerName = ageGenInputDataMap.begin()->first;

    
    // -----------------------------Prepare output blobs-----------------------------------------------------
    // Get the face network output information, set the precision, get the output node name
    auto faceOutputInfo = faceOutputDataMap.begin()->second;
    faceOutputInfo->setPrecision(Precision::FP32);
    std::string faceOutputLayerName = faceOutputDataMap.begin()->first;

    // Get pointer to data and then get the name for both outputs for the age-gender network
    auto it = ageGenOutputDataMap.begin();
    DataPtr ptrAgeOutput = (it++)->second;
    DataPtr ptrGenderOutput = (it++)->second;
    std::string ageOutputLayerName = ptrAgeOutput->getName();
    std::string genOutputLayerName = ptrGenderOutput->getName();
    // set the precision for both outputs for age gender
    for (auto& output : ageGenOutputDataMap) {
        output.second->setPrecision(Precision::FP32);
    }

    // -------------------------Loading age/gender networks to the plugin----------------------------------
    // Create executable network objects for both networks
    auto executableFaceNetwork = ie.LoadNetwork(faceNetwork, "MYRIAD");
    auto executableAgeGenNetwork = ie.LoadNetwork(ageGenNetwork, "MYRIAD");
    
    // Create inference requests for both networks
    auto faceInferRequest = executableFaceNetwork.CreateInferRequestPtr();
    auto ageGenInferRequest = executableAgeGenNetwork.CreateInferRequestPtr();

    // Set the input blobs for the inference requests
    auto faceInput = faceInferRequest->GetBlob(faceInputLayerName);
    auto ageGenInput = ageGenInferRequest->GetBlob(ageGenInputLayerName);

    // Set up buffers to be filled with input data
    auto faceInputData = faceInput->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();
    auto ageGenInputData = ageGenInput->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();

    unsigned int frame_count = 0;

    // Skip a frame after this many frames. adjust this if getting laggy camera 
    const int SKIP_AFTER = 3;
    printf("\nStarting gender_age app...\n");
    printf("\nPress any key to quit.\n");
    
    // -------------------------Running the inferences----------------------------------
    // Get the current time; inferences will only be performed periodically
    start_time = clock();
    
    // Main loop
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

        // Check if it's time to do an inference
        elapsed_time = clock() - start_time;
        if ((double)elapsed_time/(double)CLOCKS_PER_SEC >= INFERENCE_INTERVAL) 
        {
            // First run through the face detection network to find faces and crop them.
            // Afterwards, run the cropped faces through the age-gender network to receive the age-gender inference results
            // Finally, display all results in a window.
            
            // clear all detection results 
            resultColor.clear();
            resultText.clear();
            detectedFaces.clear();
            
            // ------------- Face detection network -----------------
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

            // resize the input image in accordance to the face network input size
            cv::resize(imgIn, imgInput, cv::Size(faceInputHeight, faceInputWidth));            
            size_t faceImageSize = faceInputHeight * faceInputWidth;
            // set the input data for the face network. fills buffer with input.
            for (size_t pid = 0; pid < faceImageSize; ++pid) {
                for (size_t ch = 0; ch < faceChannelsNumber; ++ch) {
                    faceInputData[ch * faceImageSize + pid] = imgInput.at<cv::Vec3b>(pid)[ch];
                }
            }
            
            // Running the request synchronously 
            faceInferRequest->Infer();
            
            // Face detection output processing //
            // Get face detection results
            auto faceOutput = faceInferRequest->GetBlob(faceOutputLayerName);
            const float *detections = faceOutput->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
            // This loop will filter out low scores and crop out the higher scoring faces which will then be sent to age-gender for inference
            for (unsigned int i = 0; i < maxProposalCount; i++) 
            {
                float image_id = detections[i * objectSize + 0];
                // exit early if the detection is not a face
                if (image_id < 0) 
                {
                    //std::cout << "Only " << i << " proposals found" << std::endl;
                    break;
                }
                
                // Calculate and save the values that we need
                // These values are the confidence scores and bounding box coordinates
                float confidence = detections[i * objectSize + 2];
                int xmin = (int)(detections[i * objectSize + 3] * width);
                int ymin = (int)(detections[i * objectSize + 4] * height);
                int xmax = (int)(detections[i * objectSize + 5] * width);
                int ymax = (int)(detections[i * objectSize + 6] * height);

                // filter out low scores
                if (confidence > FACE_DET_THRESHOLD) {
                    // Make sure coordinates are do not exceed image dimensions
                    xmin = std::max(0, xmin);
                    ymin = std::max(0, ymin);
                    xmax = std::min(width, xmax);
                    ymax = std::min(height, ymax);
                    // Crop the face and save it as a mat
                    cv::Mat croppedImage = imgIn(Rect(cv::Point2f(xmin, ymin), cv::Point2f(xmax, ymax)));
                    // Helper for current detection
                    detectionResults currentDetection;
                    // Save the cropped image
                    currentDetection.croppedMat = croppedImage;
                    
                    // Save the current face location 
                    currentDetection.xmin = xmin;
                    currentDetection.ymin = ymin;
                    currentDetection.xmax = xmax;
                    currentDetection.ymax = ymax;
                    // Put the cropped face and bounding box coordinates into the detected faces vector
                    detectedFaces.push_back(currentDetection);
                }
            }
        
            // ------------- Age Gender network -----------------
            // Get number of detected faces we got from the face network. This will be the number of inferences we will make for age gender.
            int numInferAgeGen = detectedFaces.size();
            // Get the age gender input dimensions
            auto ageGenInputDims = ageGenInferRequest->GetBlob(ageGenInputLayerName)->getTensorDesc().getDims();
            unsigned int ageGenChannelsNumber = ageGenInputDims.at(1);
            unsigned int ageGenInputHeight = ageGenInputDims.at(2);
            unsigned int ageGenInputWidth = ageGenInputDims.at(3);
            
            // Get size for the input tensor
            size_t ageGenImageSize = ageGenInputHeight * ageGenInputWidth;
            
            // Resize cropped face, then set up the input for inference for age-gender
            for (int i = 0; i < numInferAgeGen; i++)
            {
                cv::resize(detectedFaces.at(i).croppedMat, detectedFaces.at(i).croppedMat, cv::Size(ageGenInputHeight, ageGenInputWidth));
                // Set the input data for the gender network. Fill the buffer with the input data.
                for (size_t pid = 0; pid < ageGenImageSize; ++pid) 
                {
                    for (size_t ch = 0; ch < ageGenChannelsNumber; ++ch) 
                    {
                        ageGenInputData[ch * ageGenImageSize + pid] = detectedFaces.at(i).croppedMat.at<cv::Vec3b>(pid)[ch];
                    }
                }
                // Run the inference for age-gender
                ageGenInferRequest->Infer();                
                // Get the result for age
                auto ageOutput = ageGenInferRequest->GetBlob(ageOutputLayerName);
                auto ageOutputData = ageOutput->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
                // Get the result for gender
                auto genOutput = ageGenInferRequest->GetBlob(genOutputLayerName);
                auto genOutputData = genOutput->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
                
                // Age-gender output processing //
                // Find the top result for age and gender
                unsigned int results_to_display = 1;
                std::vector<unsigned> ageResults;
                std::vector<unsigned> genResults;
                TopResults(results_to_display, *ageOutput, ageResults);
                TopResults(results_to_display, *genOutput, genResults);
                
                // Calculate the confidence scores for age gender
                auto ageConf = ageOutputData[ageResults[0]]*100; // for age, the confidence score is the age. ex: if the age confidence = 19.01, the age is 19.
                auto genConf = genOutputData[genResults[0]]*100; 
                
                // Determine which color the displayed text will be. 
                // PINK = Female. Blue = Male. Green = Unknown.
                if (genConf > GENDER_CONF_THRESHOLD && genResults.at(0) == FEMALE_LABEL)
                {
                    resultColor.push_back(PINK);
                }
                else if (genConf > GENDER_CONF_THRESHOLD && genResults.at(0) == MALE_LABEL)
                {
                    resultColor.push_back(BLUE);
                }
                else 
                {
                    resultColor.push_back(GREEN);
                }
                // Add the age confidence to the results text vector
                resultText.push_back("Age: " + std::to_string((int)(ageConf)));
                elapsed_time = clock() - start_time;

            }
            start_time = clock();
        }
        
        // -----------------Display the results ---------------
        // go through all of the faces that we detected and set up the display text and bounding boxes to display gender and age
        for (unsigned int i = 0; i < detectedFaces.size(); i++)
        {
            cv::putText(imgIn, resultText.at(i), cv::Point2f(detectedFaces.at(i).xmin, detectedFaces.at(i).ymin) , FONT, FONT_SIZE, resultColor.at(i), 2);
            cv::rectangle(imgIn, cv::Point2f(detectedFaces.at(i).xmin, detectedFaces.at(i).ymin), cv::Point2f(detectedFaces.at(i).xmax, detectedFaces.at(i).ymax), resultColor.at(i), 1);
        }
        
        // Show the image in the window
        imshow(WINDOW_NAME, imgIn);
        // If the user presses any key, exit the loop
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
