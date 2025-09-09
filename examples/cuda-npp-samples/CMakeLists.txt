cmake_minimum_required(VERSION 3.10)
project(CudaNppSamples LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)
enable_language(CUDA)

# Set CUDA architectures if not already set
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86 89 90)
endif()

# Add Common directory to include path
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Common/UtilNPP)

# Add subdirectories for each sample
add_subdirectory(histEqualizationNPP)
add_subdirectory(cannyEdgeDetectorNPP)
add_subdirectory(boxFilterNPP)
add_subdirectory(FilterBorderControlNPP)
add_subdirectory(watershedSegmentationNPP)
add_subdirectory(freeImageInteropNPP)

message(STATUS "Configured CUDA NPP samples build")