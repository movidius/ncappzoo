project(simple_classifier)
cmake_minimum_required(VERSION 2.8)

include_directories(
	/opt/intel/computer_vision_sdk_2018.1.265/deployment_tools/inference_engine/include
	/opt/intel/computer_vision_sdk_2018.1.265/opencv/include/opencv2
	)

set(SOURCES
    run.cpp
)

add_executable(simple_classifier ${SOURCES})

link_directories(/opt/intel/computer_vision_sdk_2018.1.265/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64 /usr/lib/x86_64-linux-gnu/)
find_library(INFERENCE_ENGINE_LIB inference_engine /opt/intel/computer_vision_sdk_2018.1.265/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64)

target_link_libraries(simple_classifier ${INFERENCE_ENGINE_LIB} opencv_core opencv_highgui opencv_imgproc dl)

set(CMAKE_CXX_FLAGS "-Wall -std=c++11")
