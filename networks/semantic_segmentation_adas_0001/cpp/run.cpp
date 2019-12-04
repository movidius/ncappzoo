/*
 * Realsense Segmentation Sample
 *
 * Contributing Authors: Tome Vang <tome.vang@intel.com>
 *
 *
 */

#include <iostream>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <inference_engine.hpp>

// window name
#define WINDOW_NAME "Semantic Segmentation Adas 0001 - press q to quit"

#define DEVICE "MYRIAD"
// Location of the segmentation network xml file
#define SEG_NETWORK_PATH "../semantic-segmentation-adas-0001.xml"

// window height and width 4:3 ratio
#define WINDOW_WIDTH 640
#define WINDOW_HEIGHT 480
#define CAP_FPS 30

using namespace InferenceEngine;

const float alpha = 0.5; 
const float beta =  1.0 - alpha;

// Segmentation class color values. Each set of BGR values correspond to a class.
// Visit https://docs.openvinotoolkit.org/2019_R1/_semantic_segmentation_adas_0001_description_semantic_segmentation_adas_0001.html for more information.
std::vector<cv::Vec3b> colors = {
    {128, 64,  128},        
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
 * Start.
 */
int main (int argc, char** argv) 
{
    char key; 
    // ----------------------------------Segmentation Network Setup----------------------------------
    
    // ----------Create inference engine core object and network object----------
    
    Core ie;
    CNNNetwork networkObj = readNetwork(SEG_NETWORK_PATH);

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
    std::string segmentationNetworkInputLayerName = netInputDataMap.begin()->first;
    
    // ----------Prepare output blobs----------
    
    // Get the network output information, set the precision to FP32, get the output node name
    auto netOutputInfo = netOutputDataMap.begin()->second;
    netOutputInfo->setPrecision(Precision::FP32);
    std::string segmentationNetworkOutputLayerName = netOutputDataMap.begin()->first;

    // ----------Create executable network object----------
    
    // Create executable network object by loading the network and specifying the NCS device    
    auto execNetwork = ie.LoadNetwork(networkObj, DEVICE);
    
    // ----------Create inference request----------
    
    auto segmentationNetworkInferenceRequest = execNetwork.CreateInferRequestPtr();
    // Get pointers to the input and output dimensions for the network
    auto segmentationNetworkInputDims = segmentationNetworkInferenceRequest->GetBlob(segmentationNetworkInputLayerName)->getTensorDesc().getDims();
    auto segmentationNetworkOutputDims = segmentationNetworkInferenceRequest->GetBlob(segmentationNetworkOutputLayerName)->getTensorDesc().getDims();
   
    std::cout << "\nStarting Semantic Segmentation app...\n";
    std::cout << "\nPress q or Q to quit.\n";

    // ----------Get network dimensions----------

    // Get segmentation network input dimensions
    unsigned int segmentationInputChannelsNumber = segmentationNetworkInputDims.at(1);
    unsigned int segmentationInputHeight = segmentationNetworkInputDims.at(2);
    unsigned int segmentationInputWidth = segmentationNetworkInputDims.at(3);

    // Get segmentation network output dimensions
    unsigned int segmentationOutputHeight = segmentationNetworkOutputDims.at(2);
    unsigned int segmentationOutputWidth = segmentationNetworkOutputDims.at(3);    

    // ----------Read image and perform some image preprocessing----------
    
    cv::Mat originalImg = cv::imread(argv[1]);
    cv::Mat imgInput;
    // Resize the input to fit the segmentation network
    cv::resize(originalImg, imgInput, cv::Size(segmentationInputWidth, segmentationInputHeight));
   
    // ----------Fill a buffer with the image data----------
    
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
    cv::Mat segmentationColorMat;
    cv::Mat finalResultMat;

    // Resize the segmentation color mat
    cv::resize(blankMat, segmentationColorMat, cv::Size(originalImg.cols, originalImg.rows));
    // Blend both the segmentation color mat and the camera color mat together
    addWeighted(segmentationColorMat, alpha, originalImg, beta, 0.0, finalResultMat);

    // ----------OpenCV window and display setup----------
    // Set up the openCV display window
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
    cv::resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT);
    cv::setWindowProperty(WINDOW_NAME, cv::WND_PROP_ASPECT_RATIO, cv::WINDOW_KEEPRATIO);
    cv::moveWindow(WINDOW_NAME, 0, 0);

    // Show the OpenCV camera color mat 
    cv::imshow(WINDOW_NAME, finalResultMat);
    // Handle key press events
    while(true)
    {
        key = cv::waitKey(1);
        if (tolower(key) == 'q')
            break;
    }
    // ----------Clean up----------
    // Close all windows
    cv::destroyAllWindows();
    std::cout << "\nFinished." << std::endl;

    return 0;
}
