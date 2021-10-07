cmake_minimum_required(VERSION 3.10)

project(face_recognition)

message("--> Project: " ${PROJECT_NAME})
find_package(OpenCV REQUIRED)

# libraries
add_subdirectory(face_lib)

# executables
add_subdirectory(face_exe)
