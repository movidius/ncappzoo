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

// time in seconds to perform an inference on the NCS
#define INFERENCE_INTERVAL 1

using namespace std;
using namespace cv;

// enable networks
bool enableGenderNetwork = true;
bool enableAgeNetwork = true;

// text colors and font
const int FONT = cv::FONT_HERSHEY_PLAIN;
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

// the thresholds which the program uses to decide the gender of a person
// default: 0.60 and above is male, 0.40 and below is female
const float MALE_GENDER_THRESHOLD = 0.60;
const float FEMALE_GENDER_THRESHOLD = 0.40;


// device setup and preprocessing variables
double networkMean[3];
double networkStd[3];
const uint32_t MAX_NCS_CONNECTED = 2;
uint32_t numNCSConnected = 0;
mvncStatus mvncStat[MAX_NCS_CONNECTED];
const int DEV_NAME_SIZE = 100;
char mvnc_dev_name[DEV_NAME_SIZE];
void* dev_handle[MAX_NCS_CONNECTED];
void* graph_handle[MAX_NCS_CONNECTED];
std::vector<std::string> categories [MAX_NCS_CONNECTED];

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
    for (int i = 0; i < MAX_NCS_CONNECTED; i ++) {

        // get device name
        mvncStat[i] = mvncGetDeviceName(i, mvnc_dev_name, DEV_NAME_SIZE);
        if (mvncStat[i] != MVNC_OK) {
            if (mvncStat[i] == MVNC_DEVICE_NOT_FOUND) {
                if (i == 0)
                    cout << "Error - Movidius NCS not found, is it plugged in?" << endl;
                numNCSConnected = i;
                break;
            }
            else {
                cout << "Error - mvncGetDeviceName failed: " << mvncStat[i] << endl;
            }
        }
        else {
            cout << "MVNC device " << i << " name: "<< mvnc_dev_name << endl;
        }

        //open device
        mvncStat[i] = mvncOpenDevice(mvnc_dev_name, &dev_handle[i]);
        if (mvncStat[i] != MVNC_OK) {
            cout << "Error - mvncOpenDevice failed: " << mvncStat[i] << endl;
        }
        else {
            cout << "Successfully opened MVNC device" << mvnc_dev_name << endl;
            numNCSConnected++;
        }
    }

    std::cout << "Num of NCS connected: " << numNCSConnected << std::endl;
    if (numNCSConnected <= 1 && enableAgeNetwork && enableGenderNetwork) {
        cout << "Both Age and Gender networks are enabled, but only one NCS device was detected." << endl;
        cout << "Please connect two NCS devices or enable only one network (Age or Gender)." << endl;
        exit(1);
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
            mvncCloseDevice(dev_handle[0]);
            exit(1);
        }

        // allocate the graph
        mvncStat[0] = mvncAllocateGraph(dev_handle[0], &graph_handle[0], gender_graph_buf, graph_len);
        if (mvncStat[0] != MVNC_OK) {
            cout << "Error - mvncAllocateGraph failed:" << mvncStat[0] << endl;
            exit(1);
        }
        else {
            cout << "Successfully Allocated Gender graph for MVNC device." << endl;
        }

    }
}

void initAgeNetwork(){

    // Setup for Age network
    if (enableAgeNetwork) {
        if (enableGenderNetwork) {
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
                cout << "Error - Could not read graph file from disk: " << age_graph_filename << endl;
                mvncCloseDevice(dev_handle[1]);
                exit(1);
            }

            // allocate the graph
            mvncStat[1] = mvncAllocateGraph(dev_handle[1], &graph_handle[1], age_graph_buf, age_graph_len);
            if (mvncStat[1] != MVNC_OK) {
                cout << "Error - mvncAllocateGraph failed: %d\n" << mvncStat[1] << endl;
                exit(1);
            }
            else {
                cout << "Successfully Allocated Age graph for MVNC device." << endl;
            }


        } else {
            // if age is the only network selected
            if (!read_stat_txt(networkMean, networkStd, AGE_CAT_STAT_DIRECTORY)) {
                cout << "Error - Failed to read stat.txt file for age network." << endl;
                exit(1);
            }

            if (!read_cat_txt(&categories[0], AGE_CAT_STAT_DIRECTORY)) {
                cout << "Error - Failed to read categories.txt file for network.\n" << endl;
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
                cout << "Error - Could not read graph file from disk:" << age_graph_filename << endl;
                mvncCloseDevice(dev_handle[0]);
                exit(1);
            }

            mvncStat[0] = mvncAllocateGraph(dev_handle[0], &graph_handle[0], age_graph_buf, age_graph_len);
            if (mvncStat[0] != MVNC_OK) {
                cout << "Error - mvncAllocateGraph failed: " <<  mvncStat[0] << endl;
                exit(1);
            }
            else {
                cout << "Successfully Allocated Age graph for MVNC device" << endl;
            }
        }
    }
}



networkResults getInferenceResults(cv::Mat inputMat, std::vector<std::string> networkCategories, mvncStatus ncsStatus, void* graphHandle) {
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

    // now convert to array of 16 bit floating point values (half precision) and
    // pass that buffer to load tensor
    ncsStatus = mvncLoadTensor(graphHandle, tensor16, NETWORK_IMAGE_HEIGHT * NETWORK_IMAGE_WIDTH * 3 * sizeof(half_float), nullptr);
    if (ncsStatus != MVNC_OK) {
        cout << "Error! - LoadTensor failed: " << ncsStatus << endl;
        networkResults error = {-1, -1, "-1", -1};
        return error;
    }

    void* result_buf;
    unsigned int res_length;
    void* user_data;
    ncsStatus = mvncGetResult(graphHandle, &result_buf, &res_length, &user_data);
    if (ncsStatus != MVNC_OK) {
        cout << "Error! - GetResult failed: " << ncsStatus << endl;
        networkResults error = {-1, -1, "-1", -1};
        return error;
    }

    res_length /= sizeof(unsigned short);

    /* make this array large enough to hold the larger result of age/gender network results */
    float result_fp32[8];

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


int main (int argc, char** argv) {
    // Opencv variables
    Mat imgIn;
    VideoCapture capture;
    Mat croppedFaceMat;
    Scalar textColor = BLACK;
    Point topLeftRect[5];
    Point bottomRightRect[5];
    Point winTextOrigin;
    CascadeClassifier faceCascade;

    vector<Rect> faces;
    String genderText;
    String ageText;
    String rectangle_text;
    clock_t start_time, elapsed_time;
    bool start_inference_timer = true;
    int key;
    networkResults currentInferenceResult;

    capture.open(CAM_SOURCE);

    // set the window attributes
    capture.set(CV_CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT);

    // create a window
    namedWindow(WINDOW_NAME, WINDOW_NORMAL);
    resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT);
    setWindowProperty(WINDOW_NAME, CV_WND_PROP_ASPECTRATIO, CV_WINDOW_KEEPRATIO);
    
    moveWindow(WINDOW_NAME, 0, 0);
    // set a point of origin for the window text
    winTextOrigin.x = 0;
    winTextOrigin.y = 20;

    // Load XML file
    faceCascade.load(XML_FILE);


    // initialize dev handle(s)
    for (int i = 0; i < MAX_NCS_CONNECTED; i++){
        dev_handle[i] = nullptr;
    }

    // if only 1 stick is availble
    if (argc == 2) {
        if (strcmp(argv[1], "gender") == 0) {
            enableAgeNetwork = false;
        }
        if (strcmp(argv[1], "age") == 0) {
            enableGenderNetwork = false;
        }
    }

    // initiailze the NCS devices and age and gender networks
    initNCS();
    initGenderNetwork();
    initAgeNetwork();


    // main loop
    while (true) {
        // feed the capture to the opencv mat
        capture >> imgIn;

        // flip the mat horizontally
        flip(imgIn, imgIn, 1);
	
        key = waitKey(1);
        // if user presses escape then exit the loop
        if (key == 27)
            break;

        // save rectangle of detected faces to the faces vector
        faceCascade.detectMultiScale(imgIn, faces, 1.1, 2, 0| CASCADE_SCALE_IMAGE, Size(30, 30) );

        // start timer for inference intervals. Will make an inference every interval. DEFAULT is 1 second.
        if (start_inference_timer) {
            start_time = clock();
            start_inference_timer = false;
        }

        // Draw a rectangle and make an inference on each face
        for(int i = 0; i < faces.size(); i++) {
            // find the top left and bottom right corners of the rectangle of each face
            topLeftRect[i].x = faces[i].x - PADDING;
            topLeftRect[i].y = faces[i].y - PADDING;
            bottomRightRect[i].x = faces[i].x + faces[i].width + PADDING;
            bottomRightRect[i].y = faces[i].y + faces[i].height + PADDING;

            // if the rectangle is within the window bounds, draw the rectangle around the person's face
            if (topLeftRect[i].x > 0 && topLeftRect[i].y > 0 && bottomRightRect[i].x < WINDOW_WIDTH && bottomRightRect[i].y < WINDOW_HEIGHT) {
                // draw a rectangle around the detected face
                rectangle(imgIn, topLeftRect[i], bottomRightRect[i], textColor, 2, 8, 0);

                elapsed_time = clock() - start_time;

                // checks to see if it is time to make inferences
                if ((double)elapsed_time/((double)CLOCKS_PER_SEC) >= INFERENCE_INTERVAL) {

                    // crop the face from the webcam feed
                    Rect croppedFaceRect(topLeftRect[i], bottomRightRect[i]);
                    // converts the cropped face rectangle into a opencv mat
                    croppedFaceMat = imgIn(croppedFaceRect);

                    // process Gender network
                    if (enableGenderNetwork) {
                        // send the cropped opencv mat to the ncs device
                        currentInferenceResult = getInferenceResults(croppedFaceMat, categories[0], mvncStat[0], graph_handle[0]);
                        // get the appropriate color and text based on the inference results
                        if (currentInferenceResult.genderConfidence >= MALE_GENDER_THRESHOLD) {
                            genderText = categories[0].front().c_str();
                            textColor = BLUE;
                        } else
                        if (currentInferenceResult.genderConfidence <= FEMALE_GENDER_THRESHOLD){
                            genderText = categories[0].back().c_str();
                            textColor = PINK;
                        } else {
                            genderText = "Unknown";
                            textColor = BLACK;
                        }
                    }

                    // process Age network
                    if (enableAgeNetwork) {
                        if (enableGenderNetwork){
                            // send the cropped opencv mat to the ncs device
                            currentInferenceResult = getInferenceResults(croppedFaceMat, categories[1], mvncStat[1], graph_handle[1]);
                        } else {
                            currentInferenceResult = getInferenceResults(croppedFaceMat, categories[0], mvncStat[0], graph_handle[0]);
                            //cout << "Predicted Age: " << ageText << endl;
                            textColor = GREEN;
                        }
                        ageText = currentInferenceResult.ageCategory;
                    }

                    // enable starting the timer again
                    start_inference_timer = true;
                }
                // prepare the gender and age text to be printed to the window
                // rectangle_text = "id: " + to_string(i) + " " + genderText + " " + ageText;
                rectangle_text = genderText + " " + ageText;
            }
            // print the age and gender text to the window
            putText(imgIn, rectangle_text, topLeftRect[i], FONT, 3, textColor, 3);
        }

        putText(imgIn,"Press ESC to exit", winTextOrigin, FONT, 2, GREEN, 2);
        // show the opencv mat in the window
        imshow(WINDOW_NAME, imgIn);

    } // end main while loop


    // close all windows
    destroyAllWindows();

    mvncDeallocateGraph(graph_handle[0]);
    mvncCloseDevice(dev_handle[0]);
    if (numNCSConnected > 1) {
        mvncDeallocateGraph(graph_handle[1]);
        mvncCloseDevice(dev_handle[1]);
    }
    return 0;
}
