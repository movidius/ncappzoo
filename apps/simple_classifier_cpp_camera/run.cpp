// Simple classifier cpp camera

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <inference_engine.hpp>
#include <vector>

using namespace InferenceEngine;

#define DEVICE "MYRIAD"

// window properties
#define WINDOW_NAME "simple_classifier_cpp_camera - Press any key to quit"
#define WINDOW_WIDTH 640
#define WINDOW_HEIGHT 480

// text properties
const int FONT = cv::FONT_HERSHEY_SIMPLEX;
const float FONT_SIZE = 0.5;
const cv::Scalar BLUE = cv::Scalar(255, 0, 0, 255);
const cv::Scalar GREEN = cv::Scalar(0, 255, 0, 255);

std::vector<std::string> labels;
const unsigned int MAX_PATH = 256;
unsigned int SKIP_AFTER = 5;

// *************************************************************************
// Read the network labels from the provided labels file
// *************************************************************************
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
    while (fgets(cat_line , 255 , cat_file) != NULL) {
        if (cat_line[strlen(cat_line) - 1] == '\n')
            cat_line[strlen(cat_line) - 1] = '\0';
        labelsVector->push_back(std::string(cat_line));
    }
    fclose (cat_file);
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


// *************************************************************************
// Entrypoint for the application
// *************************************************************************
int main(int argc, char *argv[]) {
    cv::Mat imgInput;
    unsigned int frameCount = 0;
    
    if (argc != 3) {
        std::cout << " ./simple_classifier_cpp <XML FILE> <LABELS> ";
        exit(1);
    }
    // Get all of the parameters that we need to run the inference
    std::string XML = argv[1];
    std::string BIN = XML.substr(0, XML.length()-3) + "bin";
    std::string LABELS = argv[2];
    
    cv::Mat frame;
    int key;
    cv::VideoCapture capture;

    // Set up the camera
    capture.open(0);
    capture.set(cv::CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT);

    // Set up the display window
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);
    cv::resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT);
    cv::moveWindow(WINDOW_NAME, 0, 0);
    
    // Read the labels
    getNetworkLabels(LABELS, &labels);
    
    // ----------------------- Create IE core object and read the network ----------------------- //
    // Create the inference engine core object
    Core ie_core;
    // Create a network reader and read in the network and weights
    CNNNetwork network = ie_core.ReadNetwork(XML, BIN);
    
    // ----------------------- Set up the network input ----------------------- //
    InputsDataMap inputDataMap(network.getInputsInfo());
    InputInfo::Ptr& inputInfo = inputDataMap.begin()->second;
    // Get the input node name
    std::string inputLayerName = inputDataMap.begin()->first;
    // Set precision for the input
    inputInfo->setPrecision(Precision::U8);

    // ----------------------- Set up the network output ----------------------- //
    OutputsDataMap outputDataMap(network.getOutputsInfo());
    auto outputData = outputDataMap.begin()->second;
    // Get the output node name
    std::string outputLayerName = outputDataMap.begin()->first;
    // Set precision for output
    outputData->setPrecision(Precision::FP32);
    
    // ----------------------- Load the network and create the inference request ----------------------- //
    // Load the network to the device (default: Myriad)
    auto executableNetwork = ie_core.LoadNetwork(network, DEVICE);
    // Create the inference request
    auto inferenceRequest = executableNetwork.CreateInferRequestPtr();
    
    // ----------------------- 5. Prepare the input data ----------------------- //
    // Create buffer to hold input data
    auto inputBlob = inferenceRequest->GetBlob(inputLayerName);
    auto inputData = inputBlob->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();
    
    // Get the input dimensions for the network
    auto inputDims = inferenceRequest->GetBlob(inputLayerName)->getTensorDesc().getDims();
    unsigned int inputNumberOfChannels = inputDims.at(1);
    unsigned int inputHeight = inputDims.at(2);
    unsigned int inputWidth = inputDims.at(3);
    
    while (true) {
    
        // Use OpenCV to read in an image
        
        capture >> frame;
        if (frameCount++ >= SKIP_AFTER) {
            capture >> frame;
            frameCount = 0;
        }
        // Flip the image horizontally
        cv::flip(frame, frame, 1);
        // Resize the input image in accordance to the network input size
        cv::resize(frame, imgInput, cv::Size(inputHeight, inputWidth));
        
        // Prepare to fill the buffer with the image data
        size_t imageSize = inputHeight * inputWidth;
        // Fills buffer with the image data. This data will be sent to the device for inference
        for (size_t pixelIndex = 0; pixelIndex < imageSize; ++pixelIndex) {
            for (size_t channel = 0; channel < inputNumberOfChannels; ++channel) {
                inputData[channel * imageSize + pixelIndex] = imgInput.at<cv::Vec3b>(pixelIndex)[channel];
            }
        }
        
        // ----------------------- 6. Make the inference ----------------------- //
        inferenceRequest->StartAsync();
        if (OK == inferenceRequest->Wait(IInferRequest::WaitMode::RESULT_READY)) {
            // ----------------------- 7. Process the results ----------------------- //
            // Get the inference results
            auto inferenceResults = inferenceRequest->GetBlob(outputLayerName);
            // Get all of the confidence scores. 
            auto scores = inferenceResults->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>(); 

            // Sort the results and get the number of desired top results
            std::vector<unsigned> sortedResults; // This vector will hold all of the top sorted results
            unsigned int resultsToDisplay = 1;   // How many results should return?
            getTopResults(resultsToDisplay, *inferenceResults, sortedResults);

            // Get the top result
            auto topScore = scores[sortedResults[0]] * 100;
            // Put together the result text that we will display
            std::string resultText = labels[sortedResults.at(0)] + " - " + std::to_string((int)(topScore)) + "%";
            // Determine the text size
            cv::Size textSize = cv::getTextSize(resultText, cv::FONT_HERSHEY_SIMPLEX, FONT_SIZE, 0, 0);
            // Draw a gray rectangle for text background
            cv::rectangle(frame, cv::Point2f(0,WINDOW_HEIGHT-20), cv::Point2f(WINDOW_WIDTH, WINDOW_HEIGHT), cv::Scalar(75,75,75), cv::FILLED);
            // Calculate the coordinate to print the text so that the text is centered
            int printTextWidth = (int)((WINDOW_WIDTH - textSize.width)/2);
            // Put the text in the frame
            cv::putText(frame, resultText, cv::Point2f(printTextWidth, WINDOW_HEIGHT-5), FONT, FONT_SIZE, GREEN, 1);
        }
        
        // Display the image in the window
        imshow(WINDOW_NAME, frame);
        
        // If the user presses the break key exit the loop
        key = cv::waitKey(1);
        if (key != -1) {
            break;       
        }
    }
    cv::destroyAllWindows();
    std::cout << "\n Finished.\n";   
}
