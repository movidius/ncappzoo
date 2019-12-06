/*
 * Realsense Segmentation Sample
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
#define WINDOW_NAME "Realsense Segmentation - NCS2/OpenVINO - press q to quit"
// label file
#define segmentationNetworkLabelsFile "../seg_labels.txt"

#define DEVICE "MYRIAD"
// Location of the segmentation network xml file
#define SEG_NETWORK_PATH "../semantic-segmentation-adas-0001.xml"

// window height and width 4:3 ratio
#define WINDOW_WIDTH 640
#define WINDOW_HEIGHT 480
#define CAP_FPS 30

const unsigned int MAX_PATH = 256;

using namespace InferenceEngine;

// OpenCV display constants
const int FONT = cv::FONT_HERSHEY_PLAIN;
const float FONT_SCALE = 1.5;
const int FONT_LINE_THICKNESS = 1;
const cv::Scalar RED = cv::Scalar(0, 0, 255, 255);
const cv::Scalar BLUE = cv::Scalar(255, 0, 0, 255);
const cv::Scalar GREEN = cv::Scalar(0, 255, 0, 255);
const cv::Scalar GRAY = cv::Scalar(50, 50, 50, 255);
const cv::Scalar BLACK = cv::Scalar(0, 0, 0, 255);

const float canvasSize = 120;
const float canvasTop = (canvasSize * 0.80);
const float canvasMidTop = (canvasSize * 0.60);
const float canvasMidBottom = (canvasSize * 0.40);
const float canvasBottom = (canvasSize * 0.20);

const float alpha = 0.5; 
const float beta =  1.0 - alpha;

bool videoPauseFlag = false;
float depthMap[WINDOW_WIDTH][WINDOW_HEIGHT];

// Segmentation network variables
std::vector<long unsigned int> segmentationNetworkInputDims;
std::vector<long unsigned int> segmentationNetworkOutputDims;
InferenceEngine::InferRequest::Ptr segmentationNetworkInferenceRequest;
std::string segmentationNetworkInputLayerName;
std::string segmentationNetworkOutputLayerName;
std::vector <std::string> segmentationNetworkLabels;

cv::Mat finalResultMat;
cv::Mat segmentationColorMat;

// Segmentation class color values. Each set of BGR values correspond to a class.
// Visit https://docs.openvinotoolkit.org/2019_R1/_semantic_segmentation_adas_0001_description_semantic_segmentation_adas_0001.html for more information.
std::vector<cv::Vec3b> colors = {
    {128, 64,  128},        // background
    {232, 35,  244},        // road
    {70,  70,  70},         // sidewalk
    {156, 102, 102},        // building
    {153, 153, 190},        // wall
    {153, 153, 153},        // fence
    {30,  170, 250},        // pole
    {0,   220, 220},        // traffic light
    {35,  142, 107},        // traffic sign
    {152, 251, 152},        // vegetation
    {180, 130, 70},         // terrain
    {60,  20,  220},        // sky
    {0,   0,   255},        // person
    {142, 0,   0},          // rider
    {70,  0,   0},          // car
    {100, 60,  0},          // truck
    {90,  0,   0},          // bus
    {230, 0,   0},          // train
    {32,  11,  119},        // motorcycle
    {0,   74,  111},        // bicycle
    {81,  0,   81}          // ego-vehicle
};
       


/*
 * read network labels from file
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
 * read a network from file and return a network object
 */
InferenceEngine::CNNNetwork readNetwork(std::string inputNetworkPath) 
{
    CNNNetReader network_reader;
    network_reader.ReadNetwork(inputNetworkPath);
    network_reader.ReadWeights(inputNetworkPath.substr(0, inputNetworkPath.size() - 4) + ".bin");
    network_reader.getNetwork().setBatchSize(1);
    CNNNetwork network = network_reader.getNetwork();
    return network;
}



/*
 * get the distance map for all pixels in the image using Intel Realsense camera
 */
void getDistanceMap(rs2::depth_frame RSDepthFrame, float depthMap[WINDOW_WIDTH][WINDOW_HEIGHT])
{ 
    // Check all points and record the closest distance
    for (int horizontalDepthCheck = 0; horizontalDepthCheck < WINDOW_WIDTH; horizontalDepthCheck++)
    {
        for (int verticalDepthCheck = 0; verticalDepthCheck < WINDOW_HEIGHT; verticalDepthCheck++)
        {
            // Get the distance of a point using the depth sensor
            depthMap[horizontalDepthCheck][verticalDepthCheck] = RSDepthFrame.get_distance(horizontalDepthCheck, verticalDepthCheck);
        }
    }
}


/*
 * places text in a canvas section on a OpenCV mat
 */
void placeText(std::string text, const float canvasPosition, cv::Mat &mat)
{
    // calculate text size
    cv::Size textSize = cv::getTextSize(text, FONT, FONT_SCALE, FONT_LINE_THICKNESS, 0);
    // display centered text
    cv::putText(mat, text, cv::Point2f((mat.cols - textSize.width)/2, mat.rows - canvasPosition + (textSize.height/2)), FONT, FONT_SCALE, GREEN, FONT_LINE_THICKNESS);
}


/*
 * clears a canvas section (top, midtop, midbottom, bottom)
 */
void clearCanvasSection(const float canvasPosition, cv::Mat &mat)
{
    // calculate text size
    std::string text = "some sample text";
    cv::Size textSize = cv::getTextSize(text, FONT, FONT_SCALE, FONT_LINE_THICKNESS, 0);
    // clear out a section 
    cv::rectangle(mat, cv::Point(0, mat.rows - canvasPosition-textSize.height), cv::Point(mat.cols, mat.rows - canvasPosition + (textSize.height/2)+2), GRAY, -1);
}


/*
 * add a canvas for displaying text to an OpenCV mat
 */
void addCanvasToMat(cv::Mat &currentMat)
{
    cv::Mat canvas = cv::Mat(canvasSize, currentMat.cols, CV_8UC3);
    canvas = canvas.setTo(GRAY);
    currentMat.push_back(canvas);
}


/*
 * make an inference using the semantic segmentation network
 */
void segmentationInference(cv::Mat &cameraColorMat) 
{
    // Get segmentation network input dimensions
    unsigned int segmentationInputChannelsNumber = segmentationNetworkInputDims.at(1);
    unsigned int segmentationInputHeight = segmentationNetworkInputDims.at(2);
    unsigned int segmentationInputWidth = segmentationNetworkInputDims.at(3);

    // Get segmentation network output dimensions
    unsigned int segmentationOutputHeight = segmentationNetworkOutputDims.at(2);
    unsigned int segmentationOutputWidth = segmentationNetworkOutputDims.at(3);    

    // ----------Image preprocessing----------
    cv::Mat imgInput;
    // Resize the input to fit the segmentation network
    cv::resize(cameraColorMat, imgInput, cv::Size(segmentationInputWidth, segmentationInputHeight));
   
    // Set the input blob for the inference request using the segmentation input layer name
    auto segmentationInputBlob = segmentationNetworkInferenceRequest->GetBlob(segmentationNetworkInputLayerName);
    // Set up a 8-bit unsigned int buffer to be filled with input data
    auto segmentationInputData = segmentationInputBlob->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();
    // Set the input data for the segmentation network and fills buffer with input from the openCV color mat
    size_t segmentationImageSize = segmentationInputHeight * segmentationInputWidth;
    for (size_t pid = 0; pid < segmentationImageSize; ++pid) 
    {
        for (size_t ch = 0; ch < segmentationInputChannelsNumber; ++ch) 
        {
            segmentationInputData[ch * segmentationImageSize + pid] = imgInput.at<cv::Vec3b>(pid)[ch];
        }
    }
    
    // ----------Run the inference for the segmentation network----------
    segmentationNetworkInferenceRequest->Infer();
    
    // ----------Output Postprocessing for the segmentation network----------
    
    // Get the results from the inference
    auto segmentationOutput = segmentationNetworkInferenceRequest->GetBlob(segmentationNetworkOutputLayerName);
    const float *classDetections = segmentationOutput->buffer().as<float*>();
   
    // Create a opencv mat based on the output of the segmentation network
    // Start off with a blank opencv mat
    cv::Mat blankMat = cv::Mat(segmentationOutputHeight, segmentationOutputWidth, CV_8UC3, cv::Scalar(0,0,0));
    // Visit each of the pixels for the blank opencv Mat and determine the color values that should be there. 
    // The color values are based on the inference results from classDetections.
    for(unsigned int y = 0; y < segmentationOutputHeight; y++)
    {
        for(unsigned int x = 0; x < segmentationOutputWidth; x++)
        {
            // Get the class index from classDetections for a specific coorindate 
            int classIndexFromDetections = classDetections[y * segmentationOutputWidth + x];
            // Set the color for the blank opencv mat
            blankMat.at<cv::Vec3b>(cv::Point(x,y)) = colors.at(classIndexFromDetections);
        }
    }
    
    // Resize the segmentation color mat
    cv::resize(blankMat, segmentationColorMat, cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT));
    // Blend both the segmentation color mat and the camera color mat together
    addWeighted(cameraColorMat, alpha, segmentationColorMat, beta, 0.0, finalResultMat);
    
    // Add canvas to the final result mat
    addCanvasToMat(finalResultMat);
    
}


/*
 * handle mouse events for when the video is paused
 */
void mouseCallBackFunctionPaused(int mouseEvent, int mouseXCoordinate, int mouseYCoordinate, int flags, void* userParam)
{
   // Unpause if video is paused and the left mouse button is pressed
    if (mouseEvent == cv::EVENT_LBUTTONDOWN && videoPauseFlag == true)
    {
        videoPauseFlag = false;
    }
    
    // Handles mouse cursor move events while the video stream is paused. Reports object distance and object classification when mouse cursor moves.
    else if (mouseEvent == cv::EVENT_MOUSEMOVE && videoPauseFlag == true)
    {
        // First we wil clear the top and upper middle sections of the canvas
        clearCanvasSection(canvasTop, finalResultMat);
        clearCanvasSection(canvasMidTop, finalResultMat);
        
        // Next, we determine the distance at the mouse cursor then place the distance text on the canvas
        std::string distanceText = "Distance is ";
        if (mouseYCoordinate < WINDOW_HEIGHT)
            distanceText = distanceText + cv::format("%2.2f", depthMap[mouseXCoordinate][mouseYCoordinate]) + " meters.";
        placeText(distanceText, canvasMidTop, finalResultMat);

        // Finally, we will determine the object at the mouse cursor then place the object text on the canvas.
        // To do this, we will look at the color image created from the segmentation results.
        // We will attempt to match the color values at the mouse cursor from the segmentation color image to 
        // the ones in the colors vector (defined on line 92).
        // The indexes of the colors vector correspond to the index of a class label for the network and we can determine 
        // the object at the mouse cursor by doing this color comparision.
        std::string objectText = "Object is ";
        // Get the current color values at the current mouse coordinate
        cv::Vec3b currentColor = segmentationColorMat.at<cv::Vec3b>(mouseYCoordinate, mouseXCoordinate);
        // Create an iterator so we can iterate through our colors vector and see if there is a match 
        std::vector<cv::Vec3b>::iterator vectorIterator;
        vectorIterator = std::find(colors.begin(), colors.end(), currentColor);
        // Check to see if we found a color match and determine the object label
        if (vectorIterator != colors.end()) 
        { 
            objectText = objectText + segmentationNetworkLabels[vectorIterator - colors.begin() + 1];
        }
        placeText(objectText, canvasTop, finalResultMat);
        imshow(WINDOW_NAME, finalResultMat);
        cv::waitKey(1);
    }

}


/*
 * handle mouse events for when the video is playing
 */
void mouseCallBackFunctionPlay(int mouseEvent, int mouseXCoordinate, int mouseYCoordinate, int flags, void* cameraColorMatPtr)
{
    char key;
    // Convert the user parameter (OpenCV camera color mat) from void pointer to openCV mat
    cv::Mat cameraColorMat = *((cv::Mat*)cameraColorMatPtr);
    // Make a copy of the camera color mat and add a canvas to it
    cv::Mat cameraColorMatWithCanvas = cameraColorMat;
    addCanvasToMat(cameraColorMatWithCanvas);

    // This if-block handles left mouse-button clicks. Will "pause" the video feed.
    if (mouseEvent == cv::EVENT_LBUTTONDOWN && videoPauseFlag == false)
    {
        // Clear a section of the canvas and show loading text on the canvas
        clearCanvasSection(canvasMidTop, cameraColorMatWithCanvas);
        std::string loadingText = "Loading...";
        placeText(loadingText, canvasMidTop, cameraColorMatWithCanvas);
        imshow(WINDOW_NAME, cameraColorMatWithCanvas);
        cv::waitKey(1);
        
        videoPauseFlag = true;
        
        // Perform the inference for the segmentation network
        segmentationInference(cameraColorMat);
        
        // Show some text on the canvas 
        std::string pauseText = "Paused - Move mouse around to explore.";
        placeText(pauseText, canvasMidBottom, finalResultMat);
        std::string helpText = "Click the screen to continue.";
        placeText(helpText, canvasBottom, finalResultMat);
        imshow(WINDOW_NAME, finalResultMat);
        
        // This loop will show depth and object label when the user moves their mouse.
        while(videoPauseFlag == true)
        {
            key = cv::waitKey(1);
            if (tolower(key) == 'q')
                break;
            // Handle mouse events while paused
            cv::setMouseCallback(WINDOW_NAME, mouseCallBackFunctionPaused, NULL);
        }
    }   
}


/*
 * perform some network initialization then start the Realsense camera
 */
void initializationThenStartCamera(rs2::pipeline RSpipe)
{
    char key; 
    cv::Mat cameraColorMat;

    // ----------------------------------Segmentation Network Setup----------------------------------
    getNetworkLabels(segmentationNetworkLabelsFile, &segmentationNetworkLabels);
    
    // Create the inference engine object
    Core ie;
            
    CNNNetwork networkObj;
    // Read the network from the xml file and create a network object
    networkObj = readNetwork(SEG_NETWORK_PATH);

    // Get the input layer nodes and check to see if there are multiple inputs and outputs.
    // This sample only supports networks with 1 input and 1 output
    InputsDataMap netInputDataMap(networkObj.getInputsInfo());
    OutputsDataMap netOutputDataMap(networkObj.getOutputsInfo());
    if (netInputDataMap.size() != 1 && netOutputDataMap.size() != 1)
        std::cout << "This sample only supports segmentation networks with 1 input and 1 output" << '\n';

    // ----------Prepare input blobs----------
    
    // Get the network input information, set the precision to 8 bit unsigned int, get the network input node name
    InputInfo::Ptr& netInputInfo = netInputDataMap.begin()->second;
    netInputInfo->setPrecision(Precision::U8);
    segmentationNetworkInputLayerName = netInputDataMap.begin()->first;
    // ----------Prepare output blobs----------
    
    // Get the network output information, set the precision to FP32, get the output node name
    auto netOutputInfo = netOutputDataMap.begin()->second;
    netOutputInfo->setPrecision(Precision::FP32);
    segmentationNetworkOutputLayerName = netOutputDataMap.begin()->first;

    // ----------Loading network to the plugin----------
    
    // Create executable network object by loading the network and specifying the NCS device
    auto execNetwork = ie.LoadNetwork(networkObj, DEVICE);
    
    // ----------Create inference request and prep network input blob----------
    
    // Create inference request for the network
    segmentationNetworkInferenceRequest = execNetwork.CreateInferRequestPtr();
    // Get pointers to the input and output dimensions for the network
    segmentationNetworkInputDims = segmentationNetworkInferenceRequest->GetBlob(segmentationNetworkInputLayerName)->getTensorDesc().getDims();
    segmentationNetworkOutputDims = segmentationNetworkInferenceRequest->GetBlob(segmentationNetworkOutputLayerName)->getTensorDesc().getDims();
   
    std::cout << "\nStarting Realsense distance detection app...\n";
    std::cout << "\nPress q or Q to quit.\n";

    // ----------Main loop----------
    while (cv::getWindowProperty(WINDOW_NAME, cv::WND_PROP_AUTOSIZE) >= 0) 
    {
        // Wait for frames from Realsense camera
        rs2::frameset RSdata = RSpipe.wait_for_frames(); 
        // Get a Realsense color frame from the frame data
        auto RSColorFrame = RSdata.get_color_frame();
        auto RSDepthFrame = RSdata.get_depth_frame();
                // First we get the distance map for the current frame using the RS depth frame
        getDistanceMap(RSDepthFrame, depthMap);
        
        // Convert the Realsense color frame to an OpenCV color mat
        cameraColorMat = cv::Mat(cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT), CV_8UC3, (void*)RSColorFrame.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat cameraColorMatWithCanvas = cameraColorMat;
        
        // Add an empty "canvas" area to the OpenCV mat for displaying text
        addCanvasToMat(cameraColorMatWithCanvas);

        // Place some text in the canvas
        std::string playText = "Click anywhere to create snapshot.";
        placeText(playText, canvasTop, cameraColorMatWithCanvas);
        // Place some text in the canvas
        std::string quitText = "Press q to quit.";
        placeText(quitText, canvasMidTop, cameraColorMatWithCanvas);
        
        // Handle mouse events (this is where the inferences will happen) 
        cv::setMouseCallback(WINDOW_NAME, mouseCallBackFunctionPlay, &cameraColorMat);

        // Show the OpenCV camera color mat 
        cv::imshow(WINDOW_NAME, cameraColorMatWithCanvas);
        // Handle key press events
        key = cv::waitKey(1);
        if (tolower(key) == 'q')
            break;
    }
}


/*
 * Start.
 */
int main (int argc, char** argv) 
{
    // Set up the openCV display window
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
    cv::resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT + canvasSize);
    cv::setWindowProperty(WINDOW_NAME, cv::WND_PROP_ASPECT_RATIO, cv::WINDOW_KEEPRATIO);
    cv::moveWindow(WINDOW_NAME, 0, 0);

    // ----------Realsense setup----------
    // Create Realsense pipeline and config
    rs2::pipeline RSpipe;
    rs2::config RSconfig;
    // Enable both color and depth streams in the configuration
    RSconfig.enable_stream(RS2_STREAM_COLOR, WINDOW_WIDTH, WINDOW_HEIGHT, RS2_FORMAT_BGR8, CAP_FPS);
    RSconfig.enable_stream(RS2_STREAM_DEPTH, WINDOW_WIDTH, WINDOW_HEIGHT, RS2_FORMAT_Z16, CAP_FPS);
    // Start pipeline with config settings
    RSpipe.start(RSconfig);

    // start the camera and inferences
    initializationThenStartCamera(RSpipe);
 
    // Close all windows
    cv::destroyAllWindows();
    std::cout << "\nFinished." << std::endl;

    return 0;
}
