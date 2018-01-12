project(gender_age_lbp)
cmake_minimum_required(VERSION 3.2)
aux_source_directory(. SRC_LIST)

find_package(OpenCV 3.1.0)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(OPENCV_LIBRARIES opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_videoio opencv_objdetect)

set(PROJECT_LINK_LIBS
    ${OPENCV_LIBRARIES}
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -std=c++11 -Wall -fmessage-length=0 -pthread -fPIC -std=c++0x -fexceptions -frtti -ffunction-sections -fdata-sections")

include_directories(
    ${INCLUDE_DIR}

)

set(SOURCES 
    cpp/gender_age_lbp.cpp
    cpp/fp16.cpp
)

link_directories(
    /usr/lib)

add_executable(${PROJECT_NAME} ${SRC_LIST} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${PROJECT_LINK_LIBS} mvnc)


