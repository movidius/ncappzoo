/*
 * Realsense person distance detection
 *
 * Contributing Authors: Tome Vang <tome.vang@intel.com>
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
#include <librealsense2/rs.hpp>

#include <inference_engine.hpp>

#define WINDOW_NAME "Ncappzoo Realsense Person Distance Detection - OpenVINO"

#define labels_file "../labels.txt"
#define PERSON_LABEL_INDEX 15

// Location of ssd mobilenet network
#define SSD_NETWORK_PATH "../mobilenet-ssd.xml"


// window height and width 4:3 ratio
#define WINDOW_WIDTH 640
#define WINDOW_HEIGHT 480

const unsigned int MAX_PATH = 256;

using namespace InferenceEngine;

// text colors and font
const int FONT = cv::FONT_HERSHEY_PLAIN;
const int FONT_SIZE = 2;
const cv::Scalar RED = cv::Scalar(0, 0, 255, 255);
const cv::Scalar BLUE = cv::Scalar(255, 0, 0, 255);
const cv::Scalar GREEN = cv::Scalar(0, 255, 0, 255);

// detection thresholds and constants
const float DETECTION_THRESHOLD = 0.65;


// time to wait between ssd detection/inferences
const double INFERENCE_INTERVAL = 0.03;


// detection struct to hold results
struct detectionResults{
    float xmin = 0.0;      // coordinate of bounding box
    float ymin = 0.0;      // coordinate of bounding box
    float xmax = 0.0;      // coordinate of bounding box
    float ymax = 0.0;      // coordinate of bounding box
    std::string label = "None";
    float distance = 0.0;

};


void getNetworkLabels(std::string labelsDir, std::vector<std::string>* labelsVector)
{
    char filename[MAX_PATH];
    strncpy(filename, labelsDir.c_str(), MAX_PATH);
    FILE* cat_file = fopen(filename, "r");
    if (cat_file == nullptr) {
        std::cerr << "Could not find Category file." << std::endl;
        exit(1);
    }

    char cat_line[255];
    //fgets(cat_line , 100 , cat_file); // skip the first line
    while (fgets(cat_line , 255 , cat_file) != NULL) {
        if (cat_line[strlen(cat_line) - 1] == '\n')
            cat_line[strlen(cat_line) - 1] = '\0';
        labelsVector->push_back(std::string(cat_line));
    }
    fclose (cat_file);
}



/*
 * read a network
 */
InferenceEngine::CNNNetwork readNetwork(std::string inputNetworkPath) {
    CNNNetReader network_reader;
    network_reader.ReadNetwork(inputNetworkPath);
    network_reader.ReadWeights(inputNetworkPath.substr(0, inputNetworkPath.size() - 4) + ".bin");
    network_reader.getNetwork().setBatchSize(1);
    CNNNetwork network = network_reader.getNetwork();
    return network;
}




// Convert rs2::frame to cv::Mat
cv::Mat frame_to_mat(const rs2::frame& f)
{
    using namespace cv;
    using namespace rs2;

    auto vf = f.as<video_frame>();
    const int w = vf.get_width();
    const int h = vf.get_height();

    if (f.get_profile().format() == RS2_FORMAT_BGR8)
    {
        return Mat(Size(w, h), CV_8UC3, (void*)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_RGB8)
    {
        auto r = Mat(Size(w, h), CV_8UC3, (void*)f.get_data(), Mat::AUTO_STEP);
        cvtColor(r, r, COLOR_RGB2BGR);
        return r;
    }
    else if (f.get_profile().format() == RS2_FORMAT_Z16)
    {
        return Mat(Size(w, h), CV_16UC1, (void*)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_Y8)
    {
        return Mat(Size(w, h), CV_8UC1, (void*)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_DISPARITY32)
    {
        return Mat(Size(w, h), CV_32FC1, (void*)f.get_data(), Mat::AUTO_STEP);
    }

    throw std::runtime_error("Frame format is not supported yet!");
}



/*
 * Start.
 */
int main (int argc, char** argv) {
    //
    std::vector<std::string> labels;
    std::vector <detectionResults> detectedObjects; // vector used to hold results
    std::vector <cv::Scalar> resultColor;
    std::vector <std::string> resultText;

    // Times for inference timer
    clock_t start_time, elapsed_time;

    const int cap_width  = WINDOW_WIDTH;
    const int cap_height = WINDOW_HEIGHT;

    // Set up the display window
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
    cv::resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT);
    cv::setWindowProperty(WINDOW_NAME, cv::WND_PROP_ASPECT_RATIO, cv::WINDOW_KEEPRATIO);
    cv::moveWindow(WINDOW_NAME, 0, 0);

    getNetworkLabels(labels_file, &labels);

	// Create the inference engine object from the inference engine core api
    Core ie;

    // -------------------------Read network and check network inputs-----------------------------------------------------------
    // Declare the networks
    CNNNetwork ssdNetwork;

    // Read the network from the xml file
    ssdNetwork = readNetwork(SSD_NETWORK_PATH);

    
    // Check network input for ssd detection
    InputsDataMap ssdInputDataMap(ssdNetwork.getInputsInfo());
    OutputsDataMap ssdOutputDataMap(ssdNetwork.getOutputsInfo());
    if (ssdInputDataMap.size() != 1 && ssdOutputDataMap.size() != 1)
        throw std::logic_error("Sample supports clean SSD network with one input and one output");


    // -----------------------------Prepare input blobs-----------------------------------------------------
    // Get the ssd network input information, set the precision, get the ssd network input node name
    InputInfo::Ptr& ssdInputInfo = ssdInputDataMap.begin()->second;
    ssdInputInfo->setPrecision(Precision::U8);
    std::string ssdInputLayerName = ssdInputDataMap.begin()->first;
    

    
    // -----------------------------Prepare output blobs-----------------------------------------------------
    // Get the ssd network output information, set the precision, get the output node name
    auto ssdOutputInfo = ssdOutputDataMap.begin()->second;
    ssdOutputInfo->setPrecision(Precision::FP32);
    std::string ssdOutputLayerName = ssdOutputDataMap.begin()->first;


    // -------------------------Loading ssd network to the plugin----------------------------------
    // Create executable network objects for both networks
    auto exec_network = ie.LoadNetwork(ssdNetwork, "MYRIAD");

    
    // Create inference requests for both networks
    auto ssdInferRequest = exec_network.CreateInferRequestPtr();

    // Set the input blobs for the inference requests
    auto ssdInput = ssdInferRequest->GetBlob(ssdInputLayerName);

    // Set up buffers to be filled with input data
    auto ssdInputData = ssdInput->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();


    printf("\nStarting Realsense person distance detection app...\n");
    printf("\nPress any key to quit.\n");
    
    // -------------------------Running the inferences----------------------------------
    // Get the current time; inferences will only be performed periodically
    start_time = clock();
    
    // Realsense pipeline and config
    rs2::pipeline pipe;
  	rs2::config config;
  	// Enable both color and depth stream
    config.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 30);
    config.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    // Start pipeline with config settings
    pipe.start(config);
    
    // Main loop
    while (cv::waitKey(1) < 0 && cv::getWindowProperty(WINDOW_NAME, cv::WND_PROP_AUTOSIZE) >= 0) 
    {
        // Wait for frames from camera
        rs2::frameset data = pipe.wait_for_frames(); 
        // Get a Color and a Depth frame
        auto color_frame = data.get_color_frame();
        auto depth_frame = data.get_depth_frame();
        
        //const int d_w = depth_frame.as<rs2::video_frame>().get_width();
        //const int d_h = depth_frame.as<rs2::video_frame>().get_height();
        //const int c_w = color_frame.as<rs2::video_frame>().get_width();
        //const int c_h = color_frame.as<rs2::video_frame>().get_height();
        
        // convert the color frame to an OpenCV color mat
        cv::Mat color_mat = frame_to_mat(color_frame);
        cv::Mat flipped_color_mat;

        // Flip the image horizontally to achieve mirror effect
        cv::flip(color_mat, flipped_color_mat, 1);

        // Check if it's time to do an inference
        elapsed_time = clock() - start_time;
        if ((double)elapsed_time/(double)CLOCKS_PER_SEC >= INFERENCE_INTERVAL) 
        {
            // Prepare SSD network 
            
            // Clear all detection results 
            detectedObjects.clear();
            
            // ------------- SSD Mobilenet object detection network -----------------
            // Get pointers to the input and output dimensions for the Ssd mobilenet network
            auto ssdInputDims = ssdInferRequest->GetBlob(ssdInputLayerName)->getTensorDesc().getDims();
            auto ssdOutputDims = ssdInferRequest->GetBlob(ssdOutputLayerName)->getTensorDesc().getDims();

            // Ssd network input dimensions
            unsigned int ssdChannelsNumber = ssdInputDims.at(1);
            unsigned int ssdInputHeight = ssdInputDims.at(2);
            unsigned int ssdInputWidth = ssdInputDims.at(3);
            // Ssd network output dimensions
            unsigned int maxProposalCount = ssdOutputDims.at(2);
            unsigned int objectSize = ssdOutputDims.at(3);
            
            cv::Mat imgInput;
            // Resize the input image in accordance to the Ssd network input size
            cv::resize(flipped_color_mat, imgInput, cv::Size(ssdInputHeight, ssdInputWidth));            
            
            size_t ssdImageSize = ssdInputHeight * ssdInputWidth;
            // Set the input data for the ssd network. fills buffer with input.
            for (size_t pid = 0; pid < ssdImageSize; ++pid) {
                for (size_t ch = 0; ch < ssdChannelsNumber; ++ch) {
                    ssdInputData[ch * ssdImageSize + pid] = imgInput.at<cv::Vec3b>(pid)[ch];
                }
            }
            
            // Running the request synchronously 
            ssdInferRequest->Infer();
            
            // SSD Output Post processing //
            auto ssdOutput = ssdInferRequest->GetBlob(ssdOutputLayerName);
            const float *detections = ssdOutput->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
            // Filter out low scores
            for (unsigned int i = 0; i < maxProposalCount; i++) 
            {

                float image_id = detections[i * objectSize + 0];
                // Exit early no more objects were found
                if (image_id < 0) 
                {
                    break;
                }
                // Calculate and save the values that we need
                // These values are the confidence scores and bounding box coordinates
                int object_index = detections[i * objectSize + 1];
                float confidence = detections[i * objectSize + 2];
                int xmin = (int)(detections[i * objectSize + 3] * cap_width);
                int ymin = (int)(detections[i * objectSize + 4] * cap_height);
                int xmax = (int)(detections[i * objectSize + 5] * cap_width);
                int ymax = (int)(detections[i * objectSize + 6] * cap_height);

                // Filter out low scores and objects that do not match our desired label
                if (confidence > DETECTION_THRESHOLD && object_index == PERSON_LABEL_INDEX) {
                    // Make sure coordinates are do not exceed image dimensions
                    xmin = std::max(0, xmin);
                    ymin = std::max(0, ymin);
                    xmax = std::min(cap_width, xmax);
                    ymax = std::min(cap_height, ymax);

                    // Helper for current detection
                    detectionResults currentDetection;
                    // Calculate distance to center of the bounding box
                    float distance_to_center = depth_frame.get_distance(((float)(xmax-xmin) / 2.0 + xmin), ((float)(ymax-ymin) / 2.0 + ymin));
                    //std::cout << "Distance to center: " << distance_to_center << "\r";
                    // Save the current bounding box location 
                    currentDetection.xmin = xmin;
                    currentDetection.ymin = ymin;
                    currentDetection.xmax = xmax;
                    currentDetection.ymax = ymax;
                    currentDetection.label = labels[object_index];
                    currentDetection.distance = distance_to_center;
                    
                    // Put the bounding box coordinates, label, and distance into the detected objects vector
                    detectedObjects.push_back(currentDetection);
                }
            }
        

            // -----------------Display the results ---------------
            for (unsigned int i = 0; i < detectedObjects.size(); i++)
            {
                // Draw a red circle to indicate depth sensor focus point
                cv::circle(flipped_color_mat, cv::Point2f((detectedObjects.at(i).xmax - detectedObjects.at(i).xmin) /2 + detectedObjects.at(i).xmin, (detectedObjects.at(i).ymax - detectedObjects.at(i).ymin) /2 + detectedObjects.at(i).ymin), 2, RED, 1);
                // Display text on the bounding box
                cv::putText(flipped_color_mat, detectedObjects.at(i).label + " " + std::to_string(detectedObjects.at(i).distance), cv::Point2f(detectedObjects.at(i).xmin, detectedObjects.at(i).ymin) , FONT, FONT_SIZE, GREEN, 2);
                // Draw bounding box
                cv::rectangle(flipped_color_mat, cv::Point2f(detectedObjects.at(i).xmin, detectedObjects.at(i).ymin), cv::Point2f(detectedObjects.at(i).xmax, detectedObjects.at(i).ymax), GREEN, 1);
            }
        
            // Show the image in the window
            cv::imshow(WINDOW_NAME, flipped_color_mat);
            
        }
         
    } 

    // Close all windows
    cv::destroyAllWindows();
    std::cout << "\nFinished." << std::endl;

    return 0;
}
