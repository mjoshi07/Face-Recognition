cmake_minimum_required(VERSION 3.10)
project( face_exe )

message("--> Project: " ${PROJECT_NAME})

set(EXE_NAME "${PROJECT_NAME}")

find_package( OpenCV REQUIRED )
find_package( dlib REQUIRED )

include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIR}
        ${face_lib_INCLUDE_DIR}
)

file(GLOB_RECURSE HEADER_FILES include/*.h include/*.hpp)
file(GLOB_RECURSE SOURCES src/*.c src/*.cpp src/*.h src/*.hpp)

add_executable(${EXE_NAME} ${SOURCES} ${HEADER_FILES} )

set_target_properties(
        ${EXE_NAME}
        PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        LINKER_LANGUAGE CXX
)

target_link_libraries(
        ${EXE_NAME}
        face_lib
    )
