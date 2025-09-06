cmake_minimum_required(VERSION 3.18)
project(TestOpenNPPInstall LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找OpenNPP包
find_package(OpenNPP REQUIRED)

# 创建测试可执行文件
add_executable(test_install test_install.cpp)

# 链接OpenNPP库
target_link_libraries(test_install OpenNPP::npp)

# 设置CUDA属性
set_target_properties(test_install PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)