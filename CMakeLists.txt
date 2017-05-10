cmake_minimum_required(VERSION 3.7)
project(LabPrimitive)

set(CMAKE_CXX_STANDARD 11)

find_package( OpenCV REQUIRED )
set(SOURCE_FILES
        src/main.cpp)

add_executable(LabPrimitive ${SOURCE_FILES})
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( LabPrimitive ${OpenCV_LIBS} )