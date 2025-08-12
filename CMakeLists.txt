cmake_minimum_required(VERSION 3.18)
project(OpenNPP VERSION 1.0.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Enable CUDA language
enable_language(CUDA)

# Set CUDA architecture (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/api)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/test/framework)


# Source files
set(CPU_SOURCES
    src/modules/npp_image_arithmetic_cpu.cpp
)

set(CUDA_SOURCES
    src/modules/npp_image_arithmetic_cuda.cu
)

set(TEST_FRAMEWORK_SOURCES
    test/framework/npp_test_framework.h
    test/framework/test_nppi_addc_8u_c1rsfs.h
    test/framework/test_report.h
)

# Create static library for our NPP implementation
add_library(open_npp STATIC
    ${CPU_SOURCES}
    ${CUDA_SOURCES}
)

# Link CUDA libraries
target_link_libraries(open_npp 
    PRIVATE 
    CUDA::cudart
    CUDA::cuda_driver
)

# Set compile flags
target_compile_features(open_npp PUBLIC cxx_std_17)
target_compile_options(open_npp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr>
)

# Create test executables
add_executable(test_nppi_addc test/test_nppi_addc_8u_c1rsfs.cpp ${CPU_SOURCES})
target_link_libraries(test_nppi_addc PRIVATE CUDA::cudart)
target_compile_features(test_nppi_addc PRIVATE cxx_std_17)
target_compile_options(test_nppi_addc PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

add_executable(test_nppi_addc_validation test/test_nppi_addc_8u_c1rsfs_validation.cpp ${CPU_SOURCES} ${CUDA_SOURCES})
target_link_libraries(test_nppi_addc_validation PRIVATE CUDA::cudart)
target_compile_features(test_nppi_addc_validation PRIVATE cxx_std_17)
target_compile_options(test_nppi_addc_validation PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

# Main test runner
add_executable(test_runner test/test_runner.cpp ${CPU_SOURCES} ${CUDA_SOURCES})
target_link_libraries(test_runner PRIVATE 
    CUDA::cudart
)
target_compile_features(test_runner PRIVATE cxx_std_17)
target_compile_options(test_runner PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)


target_link_libraries(test_nppi_addc PRIVATE open_npp)
target_link_libraries(test_nppi_addc_validation PRIVATE open_npp)
target_link_libraries(test_runner PRIVATE open_npp)


# Optional: Link against actual NPP for validation
find_library(NPP_LIBRARY nppial HINTS ${CUDAToolkit_LIBRARY_DIR})
if(NPP_LIBRARY)
    message(STATUS "Found NVIDIA NPP library: ${NPP_LIBRARY}")
    target_link_libraries(test_nppi_addc_validation PRIVATE ${NPP_LIBRARY})
    target_link_libraries(test_runner PRIVATE ${NPP_LIBRARY})
    target_compile_definitions(test_nppi_addc_validation PRIVATE HAVE_NVIDIA_NPP=1)
    target_compile_definitions(test_runner PRIVATE HAVE_NVIDIA_NPP=1)
else()
    message(WARNING "NVIDIA NPP library not found - validation tests will skip NVIDIA comparison")
    target_compile_definitions(test_nppi_addc_validation PRIVATE HAVE_NVIDIA_NPP=0)
    target_compile_definitions(test_runner PRIVATE HAVE_NVIDIA_NPP=0)
endif()

# Compiler-specific options
if(MSVC)
    target_compile_options(open_npp PRIVATE /W4)
    target_compile_options(test_runner PRIVATE /W4)
else()
    target_compile_options(open_npp PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_runner PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Enable testing
enable_testing()
add_test(NAME test_nppi_addc COMMAND test_nppi_addc)
add_test(NAME test_nppi_addc_validation COMMAND test_nppi_addc_validation)
add_test(NAME test_runner COMMAND test_runner)