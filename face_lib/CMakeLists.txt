cmake_minimum_required(VERSION 3.10)
project( face_lib )

message("--> Project: " ${PROJECT_NAME})

set(LIB_NAME "${PROJECT_NAME}")

find_package(OpenCV REQUIRED)
find_package( dlib REQUIRED )

include_directories(
	${PROJECT_SOURCE_DIR}/include
	${OpenCV_INCLUDE_DIR}
)

file(GLOB_RECURSE HEADER_FILES include/*.h include/*.hpp)
file(GLOB_RECURSE SOURCES src/*.c src/*.cpp)

add_library(${LIB_NAME} STATIC ${SOURCES} ${HEADER_FILES})

set_target_properties(
        ${LIB_NAME}
        PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        LINKER_LANGUAGE CXX
)

target_link_libraries(
        ${LIB_NAME}
        ${OpenCV_LIBS}
	dlib::dlib
)

set(${PROJECT_NAME}_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include CACHE INTERNAL "${PROJECT_NAME}: Include Directories" FORCE)
