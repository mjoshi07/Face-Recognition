cmake_minimum_required(VERSION 3.10)

project(face_recognition)

option( USE_CUSTOM_BUILD_DLIB      "" OFF )
option( USE_CUSTOM_BUILD_OPENCV    "" ON )

message("--> Project: " ${PROJECT_NAME})

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

if (USE_CUSTOM_BUILD_OPENCV)
    set(OpenCV_DIR $ENV{OPENCV_LIB_DIR})
endif ()

if (USE_CUSTOM_BUILD_DLIB)
    set(dlib_DIR $ENV{DLIB_LIB_DIR})
endif ()

# libraries
add_subdirectory(face_lib)

# executables
add_subdirectory(face_exe)
