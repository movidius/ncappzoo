/*
 * Realsense distance detection
 *
 * Contributing Authors: Tome Vang <tome.vang@intel.com>
 *
 *
 */

#include <iostream>
#include <vector>
#include <time.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <librealsense2/rs.hpp>
#include <inference_engine.hpp>

// window name
#define WINDOW_NAME "Realsense Distance Detection - NCS2/OpenVINO - press q to quit"
// label file
#define labelsFile "../labels.txt"

#define DEVICE "MYRIAD"
// Location of ssd mobilenet network
#define SSD_NETWORK_PATH "../mobilenet-ssd.xml"
const std::string SSD_XML_PATH = "../mobilenet-ssd.xml";
const std::string SSD_BIN_PATH = "../mobilenet-ssd.bin";

// window height and width 4:3 ratio
#define WINDOW_WIDTH 640
#define WINDOW_HEIGHT 480
#define CAP_FPS 30

const unsigned int MAX_PATH = 256;

using namespace InferenceEngine;

// OpenCV display constants
const int FONT = cv::FONT_HERSHEY_PLAIN;
const int FONT_SIZE = 2;
const cv::Scalar RED = cv::Scalar(0, 0, 255, 255);
const cv::Scalar BLUE = cv::Scalar(255, 0, 0, 255);
const cv::Scalar GREEN = cv::Scalar(0, 255, 0, 255);
const int LINE_THICKNESS = 1;
const unsigned int CIRCLE_RADIUS = 4;

// detection thresholds and overlay constants
const float DETECTION_THRESHOLD = 0.85;
const unsigned int DEFAULT_OVERLAY_SCALE = 10;
bool displayDistanceOverlay = false;

// time to wait between ssd detection/inferences
const double INFERENCE_INTERVAL = 0.5;


// detection struct to hold results
struct detectionResults
{
    float xmin = 0.0;      // coordinate of bounding box
    float ymin = 0.0;      // coordinate of bounding box
    float xmax = 0.0;      // coordinate of bounding box
    float ymax = 0.0;      // coordinate of bounding box
    std::string label = "None";
    float distance = 0.0;

};



/*
 * read network labels
 */
void getNetworkLabels(std::string labelsDir, std::vector<std::string>* labelsVector)
{
    char filename[MAX_PATH];
    strncpy(filename, labelsDir.c_str(), MAX_PATH);
    FILE* catFile = fopen(filename, "r");
    if (catFile == nullptr) {
        std::cerr << "Could not find Category file." << std::endl;
        exit(1);
    }

    char catLine[255];
    
    while (fgets(catLine , 255 , catFile) != NULL) {
        if (catLine[strlen(catLine) - 1] == '\n')
            catLine[strlen(catLine) - 1] = '\0';
        labelsVector->push_back(std::string(catLine));
    }
    fclose (catFile);
}



/*
 * get the distance to an object using intel realsense camera
 */
float getDistanceToObject(float xmin, float ymin, float xmax, float ymax, rs2::depth_frame RSDepthFrame, cv::Mat convertedColorMat, int numHorizontalDepthChecks, int numVerticalDepthChecks)
{
    // Initialize some values
    float currentDistance = 0;
    float currentClosestDistance = 99.0;
    float closestPointX = 0;
    float closestPointY = 0;
    float deltaX = 0;
    float deltaY = 0;
    
    // Calculate the increment values for the horizontal and vertical depth check points as well as the initial padding value    
    float boundingBoxWidth = xmax - xmin;
    float boundingBoxHeight = ymax - ymin;
    float horizontalIncrementValue = boundingBoxWidth / numHorizontalDepthChecks;
    float verticalIncrementValue = boundingBoxHeight / numVerticalDepthChecks;
    float horizontalPadding = horizontalIncrementValue/2;
    float verticalPadding = verticalIncrementValue/2;
    

    // Check all points and record the closest distance
    for (int horizontalDepthCheck = 0; horizontalDepthCheck < numHorizontalDepthChecks; horizontalDepthCheck++)
    {
        for (int verticalDepthCheck = 0; verticalDepthCheck < numVerticalDepthChecks; verticalDepthCheck++)
        {
            deltaX = horizontalIncrementValue * horizontalDepthCheck;
            deltaY = verticalIncrementValue * verticalDepthCheck;
            // Get the distance of a point using the depth sensor
            currentDistance = RSDepthFrame.get_distance(xmin + horizontalPadding + deltaX, ymin + verticalPadding + deltaY);
            // Draw a red circle to indicate a depth sensor focus point
            if (displayDistanceOverlay)
                cv::circle(convertedColorMat, cv::Point2f(xmin + horizontalPadding + deltaX, ymin + verticalPadding + deltaY), CIRCLE_RADIUS, RED, cv::FILLED);
                
            // Check if the current point is the closest point
            if (currentDistance < currentClosestDistance && currentDistance > 0)
            {
                currentClosestDistance = currentDistance;
                closestPointX = xmin + horizontalPadding + deltaX;
                closestPointY = ymin + verticalPadding + deltaY;
            }
        }
    }
    // Draw the closest point using a green circle
    if (displayDistanceOverlay)
        cv::circle(convertedColorMat, cv::Point2f(closestPointX, closestPointY), CIRCLE_RADIUS, GREEN, cv::FILLED);
    
    return currentClosestDistance;
}



/*
 * Start.
 */
int main (int argc, char** argv) 
{
    char key; 
    std::vector <std::string> labels; // vector used to hold labels
    std::vector <detectionResults> detectedObjects; // vector used to hold results
    unsigned int overlayScale = DEFAULT_OVERLAY_SCALE; // distance detection overlay scale
    
    clock_t startTime, elapsedTime;

    const int captureWidth  = WINDOW_WIDTH;
    const int captureHeight = WINDOW_HEIGHT;

    // Set up the openCV display window
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
    cv::resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT);
    cv::setWindowProperty(WINDOW_NAME, cv::WND_PROP_ASPECT_RATIO, cv::WINDOW_KEEPRATIO);
    cv::moveWindow(WINDOW_NAME, 0, 0);

    getNetworkLabels(labelsFile, &labels);

    // Create the inference engine object
    Core ie;

    // ----------------------------------Read network and check network inputs----------------------------------
    
    // Read the network from the xml and bin file
    CNNNetwork ssdNetwork = ie.ReadNetwork(SSD_XML_PATH, SSD_BIN_PATH);

    // Check network input and output size 
    InputsDataMap ssdInputDataMap(ssdNetwork.getInputsInfo());
    OutputsDataMap ssdOutputDataMap(ssdNetwork.getOutputsInfo());
    if (ssdInputDataMap.size() != 1 && ssdOutputDataMap.size() != 1)
        throw std::logic_error("Sample supports clean SSD network with one input and one output");

    // ----------------------------------Prepare input blobs----------------------------------
    
    // Get the SSD network input information, set the precision, get the SSD network input node name
    InputInfo::Ptr& ssdInputInfo = ssdInputDataMap.begin()->second;
    ssdInputInfo->setPrecision(Precision::U8);
    std::string ssdInputLayerName = ssdInputDataMap.begin()->first;
    
    // ----------------------------------Prepare output blobs----------------------------------
    
    // Get the SSD network output information, set the precision, get the output node name
    auto ssdOutputInfo = ssdOutputDataMap.begin()->second;
    ssdOutputInfo->setPrecision(Precision::FP32);
    std::string ssdOutputLayerName = ssdOutputDataMap.begin()->first;

    // ----------------------------------Loading ssd network to the plugin----------------------------------
    
    // Create executable network object
    auto execNetwork = ie.LoadNetwork(ssdNetwork, DEVICE);
    
    // ----------------------------------Create inference request and prep network input blob----------------------------------
    
    // Create inference request for the network
    auto ssdInferRequest = execNetwork.CreateInferRequestPtr();

    // Get pointers to the input and output dimensions for the SSD Mobilenet network
    auto ssdInputDims = ssdInferRequest->GetBlob(ssdInputLayerName)->getTensorDesc().getDims();
    auto ssdOutputDims = ssdInferRequest->GetBlob(ssdOutputLayerName)->getTensorDesc().getDims();

    // Get SSD network input dimensions
    unsigned int ssdChannelsNumber = ssdInputDims.at(1);
    unsigned int ssdInputHeight = ssdInputDims.at(2);
    unsigned int ssdInputWidth = ssdInputDims.at(3);
    // Get SSD network output dimensions
    unsigned int maxProposalCount = ssdOutputDims.at(2);
    unsigned int objectSize = ssdOutputDims.at(3);    
    
    // Set the input blob for the inference request
    auto ssdInputBlob = ssdInferRequest->GetBlob(ssdInputLayerName);

    // Set up a buffer to be filled with input data
    auto ssdInputData = ssdInputBlob->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();

    std::cout << "\nStarting Realsense distance detection app...\n";
    std::cout << "\nPress q or Q to quit.\n";

    // ----------------------------------Realsense setup----------------------------------    
    
    // Create Realsense pipeline and config
    rs2::pipeline RSpipe;
    rs2::config RSconfig;
      // Enable both color and depth streams in the configuration
    RSconfig.enable_stream(RS2_STREAM_COLOR, WINDOW_WIDTH, WINDOW_HEIGHT, RS2_FORMAT_BGR8, CAP_FPS);
    RSconfig.enable_stream(RS2_STREAM_DEPTH, WINDOW_WIDTH, WINDOW_HEIGHT, RS2_FORMAT_Z16, CAP_FPS);
    // Start pipeline with config settings
    RSpipe.start(RSconfig);

    // Get the current time; inferences will only be performed periodically
    startTime = clock();
    
    // Main loop
    while (cv::getWindowProperty(WINDOW_NAME, cv::WND_PROP_AUTOSIZE) >= 0) 
    {
        // Wait for frames from Realsense camera
        rs2::frameset RSdata = RSpipe.wait_for_frames(); 
        // Get color and depth frame from the frame data
        auto RSColorFrame = RSdata.get_color_frame();
        auto RSDepthFrame = RSdata.get_depth_frame();
        
        // Convert the color frame to an OpenCV color mat
        cv::Mat convertedColorMat(cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT), CV_8UC3, (void*)RSColorFrame.get_data(), cv::Mat::AUTO_STEP);

        // Check if it's time to do an inference
        elapsedTime = clock() - startTime;
        if ((double)elapsedTime/(double)CLOCKS_PER_SEC >= INFERENCE_INTERVAL) 
        {
            // Clear all detection results 
            detectedObjects.clear();
            
            // -----------Input Preprocessing----------
            
            // Resize the input image in accordance to SSD Mobilenet network input size
            cv::Mat imgInput;
            cv::resize(convertedColorMat, imgInput, cv::Size(ssdInputHeight, ssdInputWidth));            
            
            // Set the input data for the SSD network. fills buffer with input.
            size_t ssdImageSize = ssdInputHeight * ssdInputWidth;
            for (size_t pid = 0; pid < ssdImageSize; ++pid) 
            {
                for (size_t ch = 0; ch < ssdChannelsNumber; ++ch) 
                {
                    ssdInputData[ch * ssdImageSize + pid] = imgInput.at<cv::Vec3b>(pid)[ch];
                }
            }
            
            // ----------Run the inference----------
            
            ssdInferRequest->Infer();
            
            // ----------Output Postprocessing----------
            
            // Get the results
            auto ssdOutput = ssdInferRequest->GetBlob(ssdOutputLayerName);
            const float *detections = ssdOutput->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
            // Filter all low scoring results, save the information for all qualifying results
            for (unsigned int detectionIndex = 0; detectionIndex < maxProposalCount; detectionIndex++) 
            {
                float imageId = detections[detectionIndex * objectSize + 0];
                // Exit early no more objects were found
                if (imageId < 0) 
                {
                    break;
                }
                // Calculate and save the confidence scores, bounding box coordinates, and class index
                int objectLabelIndex = detections[detectionIndex * objectSize + 1];
                float confidence = detections[detectionIndex * objectSize + 2];
                int xmin = (int)(detections[detectionIndex * objectSize + 3] * captureWidth);
                int ymin = (int)(detections[detectionIndex * objectSize + 4] * captureHeight);
                int xmax = (int)(detections[detectionIndex * objectSize + 5] * captureWidth);
                int ymax = (int)(detections[detectionIndex * objectSize + 6] * captureHeight);

                // Save the information for qualifying results. Discard low scoring results.
                if (confidence > DETECTION_THRESHOLD) 
                {
                    // Make sure coordinates are do not exceed image dimensions
                    xmin = std::max(0, xmin);
                    ymin = std::max(0, ymin);
                    xmax = std::min(captureWidth, xmax);
                    ymax = std::min(captureHeight, ymax);

                    // Helper for current detection
                    detectionResults currentDetection;
                    
                    // How many depth points should we sample in bounding box (horizontal and vertizal directions)?
                    // (ratio of the the object's bounding box width and height * overlay scale to the capture width and height)
                    int numberOfHorizontalDepthChecks = (xmax - xmin) * overlayScale / captureWidth;
                    int numberOfVerticalDepthChecks = (ymax - ymin) * overlayScale / captureHeight;
                    // Get the closest distance for the object 
                    float distanceToObject = getDistanceToObject(xmin, ymin, xmax, ymax, RSDepthFrame, convertedColorMat, numberOfHorizontalDepthChecks, numberOfVerticalDepthChecks);

                    // Save the current bounding box location, label, distance
                    currentDetection.xmin = xmin;
                    currentDetection.ymin = ymin;
                    currentDetection.xmax = xmax;
                    currentDetection.ymax = ymax;
                    currentDetection.label = labels[objectLabelIndex];
                    currentDetection.distance = distanceToObject;
                    
                    // Place the bounding box coordinates, label, and distance into the detected objects vector
                    detectedObjects.push_back(currentDetection);
                }
            }
        
            // -----------------Display the results ---------------
            
            for (unsigned int objectIndex = 0; objectIndex < detectedObjects.size(); objectIndex++)
            {
                // Display label and distance text on the bounding box
                std::string textToDisplay = cv::format("%s %2.2f meters", detectedObjects.at(objectIndex).label.c_str(), detectedObjects.at(objectIndex).distance);
                cv::putText(convertedColorMat, textToDisplay, cv::Point2f(detectedObjects.at(objectIndex).xmin, detectedObjects.at(objectIndex).ymin), FONT, FONT_SIZE, GREEN, 2);
                // Draw bounding box
                cv::rectangle(convertedColorMat, cv::Point2f(detectedObjects.at(objectIndex).xmin, detectedObjects.at(objectIndex).ymin), cv::Point2f(detectedObjects.at(objectIndex).xmax, detectedObjects.at(objectIndex).ymax), GREEN, LINE_THICKNESS);
            }
        
        }
        // Show the image in the window
        cv::imshow(WINDOW_NAME, convertedColorMat);
        
        // Handle key press events
        key = cv::waitKey(1);
        if (tolower(key) == 'q')
            break;
        else if (tolower(key) == 'd' && displayDistanceOverlay == false)
        {
            displayDistanceOverlay = true;
            std::cout << "Depth detection overlay: ON\n";
            std::cout << "Red dots represent depth probe locations.\n";
            std::cout << "Green dot is the closest location.\n";
        }
        else if (tolower(key) == 'd' && displayDistanceOverlay == true)
        {
            displayDistanceOverlay = false;
            std::cout << "Depth detection overlay: OFF\n";
        }
        if (tolower(key) == 's' && overlayScale > 2)
        {
            overlayScale = overlayScale - 1;
            std::cout << "Overlay scale: " << overlayScale << '\n';
        }
        if (tolower(key) == 'a' && overlayScale < 20)
        {
            overlayScale = overlayScale + 1;
            std::cout << "Overlay scale: " << overlayScale << '\n';
        }
        
    } 

    // Close all windows
    cv::destroyAllWindows();
    std::cout << "\nFinished." << std::endl;

    return 0;
}
