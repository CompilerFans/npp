# NPP 对比测试 CMake配置
cmake_minimum_required(VERSION 3.18)

project(npp_comparison_tests LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA设置
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 70;75;80;86)

# 查找依赖
find_package(GTest REQUIRED)
find_package(CUDA REQUIRED)

# 检查NVIDIA NPP可用性
if(NOT HAVE_NVIDIA_NPP)
    message(WARNING "NVIDIA NPP not available - skipping comparison tests")
    return()
endif()

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../..  # npp根目录
    ${CMAKE_CURRENT_SOURCE_DIR}/../../api  # NPP API头文件
    ${CUDA_INCLUDE_DIRS}
)

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 -lineinfo")

# 对比测试框架源文件
set(COMPARISON_TEST_FRAMEWORK_SOURCES
    framework/dual_library_test_base.h
    framework/nvidia_comparison_test_base.h
    framework/nvidia_npp_loader.h
    framework/opennpp_loader.h
)

# 对比测试源文件
set(COMPARISON_TEST_SOURCES
    arithmetic/test_nppi_arithmetic_nvidia_comparison.cpp
)

# NPP库源文件（需要链接的实现）
set(NPP_LIB_SOURCES
    ../../src/nppcore/mppcore.cpp
    ../../src/nppi/mppi_support_functions.cpp
    ../../src/nppi/nppi_arithmetic_operations/mppi_add.cpp
    ../../src/nppi/nppi_arithmetic_operations/mppi_add.cu
    ../../src/nppi/nppi_arithmetic_operations/mppi_addc.cpp
    ../../src/nppi/nppi_arithmetic_operations/mppi_addc.cu
    ../../src/nppi/nppi_arithmetic_operations/mppi_sub.cpp
    ../../src/nppi/nppi_arithmetic_operations/mppi_sub.cu
    ../../src/nppi/nppi_arithmetic_operations/mppi_subc.cpp
    ../../src/nppi/nppi_arithmetic_operations/mppi_subc.cu
    ../../src/nppi/nppi_arithmetic_operations/mppi_mul.cpp
    ../../src/nppi/nppi_arithmetic_operations/mppi_mul.cu
    ../../src/nppi/nppi_arithmetic_operations/mppi_mulc.cpp
    ../../src/nppi/nppi_arithmetic_operations/mppi_mulc.cu
)

# 创建NPP对比测试可执行文件
add_executable(npp_comparison_tests
    ${COMPARISON_TEST_SOURCES}
    ${NPP_LIB_SOURCES}
)

# 设置CUDA分离编译
set_property(TARGET npp_comparison_tests PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# 链接库
target_link_libraries(npp_comparison_tests
    GTest::gtest
    GTest::gtest_main
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
    dl  # 动态库加载支持
)

# 添加编译定义
target_compile_definitions(npp_comparison_tests PRIVATE 
    HAVE_NVIDIA_NPP=1
    NVIDIA_NPP_IAL_PATH="${NPP_LIBRARY_IAL}"
    NVIDIA_NPP_CORE_PATH="${NPP_LIBRARY_CORE}"
)

# 添加测试
enable_testing()
add_test(NAME NPP_Comparison_Tests COMMAND npp_comparison_tests)

# 创建算术模块对比测试目标
add_executable(npp_arithmetic_comparison_tests
    arithmetic/test_nppi_arithmetic_nvidia_comparison.cpp
    ${NPP_LIB_SOURCES}
)

set_property(TARGET npp_arithmetic_comparison_tests PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(npp_arithmetic_comparison_tests
    GTest::gtest
    GTest::gtest_main
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
    dl
)

target_compile_definitions(npp_arithmetic_comparison_tests PRIVATE 
    HAVE_NVIDIA_NPP=1
    NVIDIA_NPP_IAL_PATH="${NPP_LIBRARY_IAL}"
    NVIDIA_NPP_CORE_PATH="${NPP_LIBRARY_CORE}"
)

add_test(NAME NPP_Arithmetic_Comparison_Tests COMMAND npp_arithmetic_comparison_tests)

# 安装规则（可选）
install(TARGETS npp_comparison_tests
    RUNTIME DESTINATION bin
)

# 打印配置信息
message(STATUS "NPP Comparison Tests Configuration:")
message(STATUS "  NVIDIA NPP Available: ${HAVE_NVIDIA_NPP}")
message(STATUS "  NVIDIA NPP IAL Path: ${NPP_LIBRARY_IAL}")
message(STATUS "  NVIDIA NPP CORE Path: ${NPP_LIBRARY_CORE}")
message(STATUS "  CUDA Version: ${CUDA_VERSION}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")