/*
 * Gender_Age_Lbp
 *
 * Contributing Authors: Tome Vang <tome.vang@intel.com>, Neal Smith <neal.p.smith@intel.com>, Heather McCabe <heather.m.mccabe@intel.com>
 *
 *
 *
 */

#include <iostream>
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "fp16.h"
#include <time.h>
#include <stdint.h>

extern "C"
{
#include <mvnc.h>

}

#define WINDOW_NAME "Ncappzoo Gender Age"
#define CAM_SOURCE 0
#define XML_FILE "../lbpcascade_frontalface_improved.xml"
// window height and width 16:9 ratio
#define WINDOW_WIDTH 640
#define WINDOW_HEIGHT 360

// network image resolution
#define NETWORK_IMAGE_WIDTH 227
#define NETWORK_IMAGE_HEIGHT 227

// Location of age and gender networks
#define GENDER_GRAPH_DIR "../gender_graph/"
#define AGE_GRAPH_DIR "../age_graph/"
#define GENDER_CAT_STAT_DIRECTORY "../catstat/Gender/"
#define AGE_CAT_STAT_DIRECTORY "../catstat/Age/"

using namespace std;
using namespace cv;

// enable networks
bool enableGenderNetwork = true;
bool enableAgeNetwork = true;

// text colors and font
const int FONT = cv::FONT_HERSHEY_PLAIN;
const int FONT_SIZE = 2;
const cv::Scalar BLUE = cv::Scalar(255, 0, 0, 255);
const cv::Scalar GREEN = cv::Scalar(0, 255, 0, 255);
const cv::Scalar RED = cv::Scalar(0, 0, 255, 255);
const cv::Scalar PINK = Scalar(255, 80, 180, 255);
const cv::Scalar BLACK = Scalar(0, 0, 0, 255);

// max chars to use for full path.
const unsigned int MAX_PATH = 256;

// opencv cropped face padding. make this larger to increase rectangle size
// default: 60
const int PADDING = 60;

// time to wait between face detection/inferences
const double INFERENCE_INTERVAL = 0.05;

// the thresholds which the program uses to decide the gender of a person
// default: 0.60 and above is male, 0.40 and below is female
const float MALE_GENDER_THRESHOLD = 0.60;
const float FEMALE_GENDER_THRESHOLD = 0.40;


// device setup and preprocessing variables
double networkMean[3];
double networkStd[3];
const uint32_t MAX_NCS_CONNECTED = 2;
uint32_t numNCSConnected = 0;
ncStatus_t retCode;
struct ncDeviceHandle_t* dev_handle[MAX_NCS_CONNECTED];
struct ncGraphHandle_t* age_graph_handle = NULL;
struct ncGraphHandle_t* gender_graph_handle = NULL;
struct ncFifoHandle_t* age_fifo_in = NULL;
struct ncFifoHandle_t* age_fifo_out = NULL;
struct ncFifoHandle_t* gender_fifo_in = NULL;
struct ncFifoHandle_t* gender_fifo_out = NULL;
std::vector<std::string> categories [2];

typedef unsigned short half_float;

//--------------------------------------------------------------------------------
// // struct for holding age and gender results
//--------------------------------------------------------------------------------
typedef struct networkResults {
    int gender;
    float genderConfidence;
    string ageCategory;
    float ageConfidence;
}networkResults;


bool preprocess_image(const cv::Mat& src_image_mat, cv::Mat& preprocessed_image_mat)
{
    // find ratio of to adjust width and height by to make them fit in network image width and height
    double width_ratio = (double)NETWORK_IMAGE_WIDTH / (double)src_image_mat.cols;
    double height_ratio = (double)NETWORK_IMAGE_HEIGHT / (double)src_image_mat.rows;

    // the largest ratio is the one to use for scaling both height and width.
    double largest_ratio = (width_ratio > height_ratio) ? width_ratio : height_ratio;

    // resize the image as close to the network required image dimensions.  After scaling the
    // based on the largest ratio, the resized image will still be in the same aspect ratio as the
    // camera provided but either height or width will be larger than the network required height
    // or width (unless network height == network width.)
    cv::resize(src_image_mat, preprocessed_image_mat, cv::Size(), largest_ratio, largest_ratio, CV_INTER_AREA);

    // now that the preprocessed image is resized, we'll just extract the center portion of it that is exactly the
    // network height and width.
    int mid_row = preprocessed_image_mat.rows / 2.0;
    int mid_col = preprocessed_image_mat.cols / 2.0;
    int x_start = mid_col - (NETWORK_IMAGE_WIDTH/2);
    int y_start = mid_row - (NETWORK_IMAGE_HEIGHT/2);
    cv::Rect roi(x_start, y_start, NETWORK_IMAGE_WIDTH, NETWORK_IMAGE_HEIGHT);
    preprocessed_image_mat = preprocessed_image_mat(roi);
    return true;
}


bool read_stat_txt(double* network_mean, double* network_std, const string NETWORK_DIR)
{
    char filename[MAX_PATH];
    strncpy(filename, NETWORK_DIR.c_str(), MAX_PATH);
    strncat(filename, "stat.txt", MAX_PATH);
    FILE* stat_file = fopen(filename, "r");
    if (stat_file == nullptr) {
        return false;
    }
    int num_read_std = 0;
    int num_read_mean = 0;
    num_read_mean = fscanf(stat_file, "%lf%lf%lf\n", &(network_mean[0]), &(network_mean[1]), &(network_mean[2]));
    if (num_read_mean == 3) {
        num_read_std = fscanf(stat_file, "%lf%lf%lf", &(network_std[0]), &(network_std[1]), &(network_std[2]));
    }
    fclose(stat_file);

    if (num_read_mean != 3 || num_read_std != 3) {
        return false;
    }

    for (int i = 0; i < 3; i++) {
        network_mean[i] = 255.0 * network_mean[i];
        network_std[i] = 1.0 / (255.0 * network_std[i]);
    }

    return true;
}

bool read_cat_txt(std::vector<std::string> *categories, const string NETWORK_DIR)
{
    char filename[MAX_PATH];
    strncpy(filename, NETWORK_DIR.c_str(), MAX_PATH);
    strncat(filename, "categories.txt", MAX_PATH);
    FILE* cat_file = fopen(filename, "r");
    if (cat_file == nullptr) {
        return false;
    }

    char cat_line[100];
    fgets(cat_line , 100 , cat_file); // skip the first line
    while (fgets(cat_line , 100 , cat_file) != NULL) {
        if (cat_line[strlen(cat_line) - 1] == '\n')
            cat_line[strlen(cat_line) - 1] = '\0';
        categories->push_back(std::string(cat_line));
    }
    fclose (cat_file);

    if (categories->size() < 1) {
        return false;
    }

    return true;
}

/**
 * @brief read_graph_from_file
 * @param graph_filename [IN} is the full path (or relative) to the graph file to read.
 * @param length [OUT] upon successful return will contain the number of bytes read
 *        which will correspond to the number of bytes in the buffer (graph_buf) allocated
 *        within this function.
 * @param graph_buf [OUT] should be set to the address of a void pointer prior to calling
 *        this function.  upon successful return the void* pointed to will point to a
 *        memory buffer which contains the graph file that was read from disk.  This buffer
 *        must be freed when the caller is done with it via the free() system call.
 * @return true if worked and program should continue or false there was an error.
 */
bool read_graph_from_file(const char *graph_filename, unsigned int *length_read, void **graph_buf)
{
    FILE *graph_file_ptr;

    *graph_buf = nullptr;

    graph_file_ptr = fopen(graph_filename, "rb");
    if (graph_file_ptr == nullptr) {
        return false;
    }

    // get number of bytes in file
    *length_read = 0;
    fseek(graph_file_ptr, 0, SEEK_END);
    *length_read = ftell(graph_file_ptr);
    rewind(graph_file_ptr);

    if(!(*graph_buf = malloc(*length_read))) {
        // couldn't allocate buffer
        fclose(graph_file_ptr);
        return false;
    }

    size_t to_read = *length_read;
    size_t read_count = fread(*graph_buf, 1, to_read, graph_file_ptr);

    if(read_count != *length_read) {
        // didn't read the expected number of bytes
        fclose(graph_file_ptr);
        free(*graph_buf);
        *graph_buf = nullptr;
        return false;
    }
    fclose(graph_file_ptr);

    return true;
}

/**
 * @brief compare result data to sort result indexes
 */
static float *result_data;
int sort_results(const void * index_a, const void * index_b) {
    int *a = (int *)index_a;
    int *b = (int *)index_b;
    float diff = result_data[*b] - result_data[*a];
    if (diff < 0) {
        return -1;
    } else if (diff > 0) {
        return 1;
    } else {
        return 0;
    }
}



void initNCS(){
    for (int i = 0; i < MAX_NCS_CONNECTED; i++) {
        //initialize device handles
        struct ncDeviceHandle_t* dev;
        dev_handle[i] == dev;
        retCode = ncDeviceCreate(i, &dev_handle[i]);
        if (retCode != NC_OK) {
            if (i == 0) {
                cout << "Error - No neural compute device found." << endl;
            }
            break;
        }

        //open device
        retCode = ncDeviceOpen(dev_handle[i]);
        if (retCode != NC_OK) {
            cout << "Error[" << retCode << "] - could not open device at index " << i << "." << endl;
        }
        else {
            numNCSConnected++;
        }

    }

    if (numNCSConnected > 0) {
        cout << "Num of neural compute devices connected: " << numNCSConnected << endl;
    }
}

void initGenderNetwork() {
    // Setup for Gender network
    if (enableGenderNetwork) {
        // Read the gender stat file
        if (!read_stat_txt(networkMean, networkStd, GENDER_CAT_STAT_DIRECTORY)) {
            cout << "Error - Failed to read stat.txt file for gender network." << endl;
            exit(1);
        }
        // Read the gender cat file
        if (!read_cat_txt(&categories[0], GENDER_CAT_STAT_DIRECTORY)) {
            cout << "Error - Failed to read categories.txt file for gender network." << endl;
            exit(1);
        }

        // read the gender graph from file:
        char gender_graph_filename[MAX_PATH];
        strncpy(gender_graph_filename, GENDER_GRAPH_DIR, MAX_PATH);
        strncat(gender_graph_filename, "graph", MAX_PATH);
        unsigned int graph_len = 0;
        void *gender_graph_buf;
        if (!read_graph_from_file(gender_graph_filename, &graph_len, &gender_graph_buf)) {
            // error reading graph
            cout << "Error - Could not read graph file from disk: " << gender_graph_filename << endl;
            exit(1);
        }

         // initialize the graph handle
        retCode = ncGraphCreate("genderGraph", &gender_graph_handle);

        // allocate the graph
        retCode = ncGraphAllocateWithFifosEx(dev_handle[0], gender_graph_handle, gender_graph_buf, graph_len,
                                           &gender_fifo_in, NC_FIFO_HOST_WO, 2, NC_FIFO_FP16,
                                           &gender_fifo_out, NC_FIFO_HOST_RO, 2, NC_FIFO_FP16);
        if (retCode != NC_OK) {
            cout << "Error[" << retCode << "]- could not allocate gender network." << endl;
            exit(1);
        }
        else {
            cout << "Successfully allocated gender graph to device 0." << endl;
        }

    }
}

void initAgeNetwork(){

    // Setup for Age network
    if (enableAgeNetwork) {
        // determine if this will be allocated to the first or second device
        int dev_index = 0;
        if (numNCSConnected > 1 && enableGenderNetwork) {
            // if more than one device connected and the first device was used for gender graph use second device
            dev_index = 1;
        }

        // read age stat file
        if (!read_stat_txt(networkMean, networkStd, AGE_CAT_STAT_DIRECTORY)) {
            cout << "Error - Failed to read stat.txt file for age network." << endl;
            exit(1);
        }
        // read cat txt file
        if (!read_cat_txt(&categories[1], AGE_CAT_STAT_DIRECTORY)) {
            cout << "Error - Failed to read categories.txt file for age network." << endl;
            exit(1);
        }

        // read the age graph from file:
        char age_graph_filename[MAX_PATH];
        strncpy(age_graph_filename, AGE_GRAPH_DIR, MAX_PATH);
        strncat(age_graph_filename, "graph", MAX_PATH);
        unsigned int age_graph_len = 0;
        void *age_graph_buf;
        if (!read_graph_from_file(age_graph_filename, &age_graph_len, &age_graph_buf)) {
            // error reading graph
            exit(1);
        }

        // initialize the graph handle
        retCode = ncGraphCreate("ageGraph", &age_graph_handle);

        // allocate the graph
        retCode = ncGraphAllocateWithFifosEx(dev_handle[dev_index], age_graph_handle, age_graph_buf, age_graph_len,
                                           &age_fifo_in, NC_FIFO_HOST_WO, 2, NC_FIFO_FP16,
                                           &age_fifo_out, NC_FIFO_HOST_RO, 2, NC_FIFO_FP16);
        if (retCode != NC_OK) {
            cout << "Error[" << retCode << "]- could not allocate gender network." << endl;
            exit(1);
        }
        else {
            cout << "Successfully allocated age graph to device " << dev_index << "." << endl;
        }
    }
}



networkResults getInferenceResults(cv::Mat inputMat, std::vector<std::string> networkCategories,
                                   struct ncGraphHandle_t* graphHandle, struct ncFifoHandle_t* fifoIn,
                                   struct ncFifoHandle_t* fifoOut) {
    cv::Mat preprocessed_image_mat;
    preprocess_image(inputMat, preprocessed_image_mat);
    if (preprocessed_image_mat.rows != NETWORK_IMAGE_HEIGHT ||
        preprocessed_image_mat.cols != NETWORK_IMAGE_WIDTH) {
        cout << "Error - preprocessed image is unexpected size!" << endl;
        networkResults error = {-1, -1, "-1", -1};
        return error;
    }

    // three values for each pixel in the image.  one value for each color channel RGB
    float_t tensor32[3];
    half_float tensor16[NETWORK_IMAGE_WIDTH * NETWORK_IMAGE_HEIGHT * 3];

    uint8_t* image_data_ptr = (uint8_t*)preprocessed_image_mat.data;
    int chan = preprocessed_image_mat.channels();


    int tensor_index = 0;
    for (int row = 0; row < preprocessed_image_mat.rows; row++) {
        for (int col = 0; col < preprocessed_image_mat.cols; col++) {

            int pixel_start_index = row * (preprocessed_image_mat.cols + 0) * chan + col * chan; // TODO: don't hard code

            // assuming the image is in BGR format here
            uint8_t blue = image_data_ptr[pixel_start_index + 0];
            uint8_t green = image_data_ptr[pixel_start_index + 1];
            uint8_t red = image_data_ptr[pixel_start_index + 2];

            //image_data_ptr[pixel_start_index + 2] = 254;

            // then assuming the network needs the data in BGR here.
            // also subtract the mean and multiply by the standard deviation from stat.txt file
            tensor32[0] = (((float_t)blue - networkMean[0]) * networkStd[0]);
            tensor32[1] = (((float_t)green - networkMean[1]) * networkStd[1]);
            tensor32[2] = (((float_t)red - networkMean[2]) * networkStd[2]);

            tensor16[tensor_index++] =  float2half(*((unsigned*)(&(tensor32[0]))));
            tensor16[tensor_index++] =  float2half(*((unsigned*)(&(tensor32[1]))));
            tensor16[tensor_index++] =  float2half(*((unsigned*)(&(tensor32[2]))));
        }
    }

    // queue for inference
    unsigned int inputTensorLength = NETWORK_IMAGE_HEIGHT * NETWORK_IMAGE_WIDTH * 3 * sizeof(half_float);
    retCode = ncGraphQueueInferenceWithFifoElem(graphHandle, fifoIn, fifoOut, tensor16,  &inputTensorLength, 0);
    if (retCode != NC_OK) {
        cout << "Error[" << retCode << "] - could not queue inference." << endl;
        networkResults error = {-1, -1, "-1", -1};
        return error;
    }

    // get the size of the result
    unsigned int res_length;
    unsigned int option_length = sizeof(res_length);
    retCode = ncFifoGetOption(fifoOut, NC_RO_FIFO_ELEMENT_DATA_SIZE, &res_length, &option_length);
    if (retCode != NC_OK) {
        cout << "Error[" << retCode << "] - could not get output result size." << endl;
        networkResults error = {-1, -1, "-1", -1};
        return error;
    }

    half_float result_buf[res_length];
    void* user_data;
    retCode = ncFifoReadElem(fifoOut, result_buf, &res_length, &user_data);
    if (retCode != NC_OK) {
        cout << "Error[" << retCode << "] - could not get output result." << endl;
        networkResults error = {-1, -1, "-1", -1};
        return error;
    }


    res_length /= sizeof(unsigned short);
    float result_fp32[2];
    fp16tofloat(result_fp32, (unsigned char*)result_buf, res_length);

    // Sort the results to get the top result
    int indexes[res_length];
    for (unsigned int i = 0; i < res_length; i++) {
        indexes[i] = i;
    }
    result_data = result_fp32;

    networkResults personInferenceResults;

    if (strcmp(networkCategories[indexes[0]].c_str(), "Male") == 0) {
        personInferenceResults.gender = indexes[0];
        personInferenceResults.genderConfidence = result_fp32[indexes[0]];
    }
    if (strcmp(networkCategories[indexes[0]].c_str(), "0-2") == 0) {
        qsort(indexes, res_length, sizeof(*indexes), sort_results);
        personInferenceResults.ageCategory = networkCategories[indexes[0]].c_str();
        personInferenceResults.ageConfidence = result_fp32[indexes[0]];
    }

    return personInferenceResults;

}

/*
 * Used to sort faces from left to right.
 */
bool sortFaces(Rect face1, Rect face2) { return face1.x < face2.x; }


int main (int argc, char** argv) {
    // Camera and image frames
    VideoCapture capture;
    Mat imgIn;

    // Face detection
    CascadeClassifier faceCascade;
    vector<Rect> faces;
    vector<String> resultText;
    vector<Scalar> resultColor;
    Mat croppedFaceMat;
    
    // Times for inference timer
    clock_t start_time, elapsed_time;

    // Key to escape from main loop and close program
    const int breakKey = 27;  // esc == 27
    int key;

    // Struct that will hold inference results
    networkResults currentInferenceResult;

    // Set up the camera
    capture.open(CAM_SOURCE);
    capture.set(CV_CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT);

    // Set up the display window
    namedWindow(WINDOW_NAME, WINDOW_NORMAL);
    resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT);
    setWindowProperty(WINDOW_NAME, CV_WND_PROP_ASPECTRATIO, CV_WINDOW_KEEPRATIO);
    moveWindow(WINDOW_NAME, 0, 0);
    Point winTextOrigin(0, 20);

    // Load XML file for the face detection cascade classifier
    if (!faceCascade.load(XML_FILE)) {
        cout << "Error - could not load face detection cascade XML file." << endl;
        return -1;
    }

    // Check if a specific network was provided on the command line (if so, the other won't be used)
    if (argc == 2) {
        if (strcmp(argv[1], "gender") == 0) {
            enableAgeNetwork = false;
        } else if (strcmp(argv[1], "age") == 0) {
            enableGenderNetwork = false;
        }
    }

    // Initialize the NCS device(s) and network graphs and FIFO queues
    initNCS();
    if (numNCSConnected == 0) return -1;
    initGenderNetwork();
    initAgeNetwork();

    // Get the current time; inferences will only be performed periodically
    start_time = clock();

    unsigned int frame_count = 0;

    // skip a frame after this many frames
    // adjust this if getting laggy camera 
    const int SKIP_AFTER = 3;

    // main loop
    while (true) {
        // If the user presses the break key exit the loop
        key = waitKey(1);
        if ((key & 0xFF) == breakKey) {
            break;
        }

        // Get a frame from the camera
        capture >> imgIn;
	if (frame_count++ >= SKIP_AFTER)
        {
	    capture >> imgIn;
            frame_count = 0;
        }

        // Flip the image horizontally
        flip(imgIn, imgIn, 1);

        // Check if it is time to do an inference
        elapsed_time = clock() - start_time;
        if ((double)elapsed_time/(double)CLOCKS_PER_SEC >= INFERENCE_INTERVAL) {

            // Clear the label and color vectors
            resultText.clear();
            resultColor.clear();

            // Detect faces and sort from left to right
            faceCascade.detectMultiScale(imgIn, faces, 1.1, 2, 0| CASCADE_SCALE_IMAGE, Size(30, 30) );
            sort(faces.begin(), faces.end(), sortFaces);

            // Process each face
            for(int i = 0; i < faces.size(); i++) {
                // Expand the detected face boundaries to have more padding and include the whole head
                // Or if the rectangle boundary falls outside the window cut it off at the edge
                Point topLeftCorner(max(faces[i].x - PADDING, 0), max(faces[i].y - PADDING, 0));
                Point bottomRightCorner(min(faces[i].x + faces[i].width + PADDING, WINDOW_WIDTH), min(faces[i].y + faces[i].height + PADDING, WINDOW_HEIGHT));
                faces[i] = Rect(topLeftCorner, bottomRightCorner);

                // Crop the face from the image
                //Rect croppedFaceRect(topLeftRect[i], bottomRightRect[i]);
                croppedFaceMat = imgIn(faces[i]);

                // Process the GenderNet network
                if (enableGenderNetwork) {
                    // Queue an inference and get the inference result
                    currentInferenceResult = getInferenceResults(croppedFaceMat, categories[0], gender_graph_handle,
                                                                 gender_fifo_in, gender_fifo_out);

                    // Get the correct color and gender label
                    if (currentInferenceResult.genderConfidence >= MALE_GENDER_THRESHOLD) {
                        resultText.push_back(categories[0].front().c_str());
                        resultColor.push_back(BLUE);
                    } else if (currentInferenceResult.genderConfidence <= FEMALE_GENDER_THRESHOLD){
                        resultText.push_back(categories[0].back().c_str());
                        resultColor.push_back(PINK);
                    } else {
                        resultText.push_back("Unknown ");
                        resultColor.push_back(BLACK);
                    }
                } else {
                    // Not processing for gender
                    resultText.push_back("");
                    resultColor.push_back(BLACK);
                }

                // Process the AgeNet network
                if (enableAgeNetwork) {
                    // Queue an inference and get the result
                    currentInferenceResult = getInferenceResults(croppedFaceMat, categories[1], age_graph_handle,
                                                                 age_fifo_in, age_fifo_out);
                    resultText[i] += currentInferenceResult.ageCategory;
                }
            }
            // Reset the inference timer
            start_time = clock();
        }

        // Draw labels and rectangles on the image
        putText(imgIn,"Press ESC to exit", winTextOrigin, FONT, 2, GREEN, 2);
        for(int i = 0; i < faces.size(); i++) {
            // Draw a rectangle around the detected face
            rectangle(imgIn, faces[i].tl(), faces[i].br(), resultColor[i], 2, 8, 0);

            // print the age and gender text to the window
            putText(imgIn, resultText[i], faces[i].tl(), FONT, FONT_SIZE, resultColor[i], 3);
        }

        // Show the image in the window
        imshow(WINDOW_NAME, imgIn);

    } // end main while loop


    // Close all windows
    destroyAllWindows();

    // NCAPI clean up
    ncFifoDestroy(&age_fifo_in);
    ncFifoDestroy(&age_fifo_out);
    ncFifoDestroy(&gender_fifo_in);
    ncFifoDestroy(&gender_fifo_out);
    ncGraphDestroy(&age_graph_handle);
    ncGraphDestroy(&gender_graph_handle);
    for (int i = 0; i < numNCSConnected; i++) {
        ncDeviceClose(dev_handle[i]);
        ncDeviceDestroy(&dev_handle[i]);
    }

    return 0;
}
