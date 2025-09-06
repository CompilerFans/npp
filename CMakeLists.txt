cmake_minimum_required(VERSION 3.18)
project(OpenNPP VERSION 1.0.0)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找必需包
find_package(CUDAToolkit REQUIRED)

# 启用CUDA语言
enable_language(CUDA)

# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures")

# 构建src目录 - 核心库
add_subdirectory(src)

# 构建选项
option(BUILD_TESTS "Build tests" ON)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()