project(simple_classifier_cpp)
cmake_minimum_required(VERSION 2.8)

include_directories(
	$ENV{INTEL_CVSDK_DIR}/deployment_tools/inference_engine/include
	$ENV{INTEL_CVSDK_DIR}/opencv/include
	
)

set(SOURCES
    run.cpp
)

        
add_executable(simple_classifier_cpp ${SOURCES})

link_directories($ENV{INTEL_CVSDK_DIR}/deployment_tools/inference_engine/lib/intel64 /usr/lib/x86_64-linux-gnu/)

find_library(INFERENCE_ENGINE_LIB inference_engine $ENV{INTEL_CVSDK_DIR}/deployment_tools/inference_engine/lib/intel64)
find_library(OPENCV_CORE opencv_core $ENV{INTEL_CVSDK_DIR}/opencv/lib)
find_library(OPENCV_HIGHGUI opencv_highgui $ENV{INTEL_CVSDK_DIR}/opencv/lib)
find_library(OPENCV_IMGPROC opencv_imgproc $ENV{INTEL_CVSDK_DIR}/opencv/lib)
find_library(OPENCV_CODECS opencv_imgcodecs $ENV{INTEL_CVSDK_DIR}/opencv/lib)

target_link_libraries(simple_classifier_cpp ${INFERENCE_ENGINE_LIB} ${OPENCV_CORE} ${OPENCV_HIGHGUI} ${OPENCV_IMGPROC} ${OPENCV_CODECS})

set(CMAKE_CXX_FLAGS "-Wall -std=c++11")
