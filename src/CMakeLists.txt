# NPP核心库CMake配置

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../api
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# 递归查找所有源文件
file(GLOB_RECURSE CPU_SOURCES 
    "*.cpp"
)

file(GLOB_RECURSE CUDA_SOURCES
    "*.cu"
)

# 创建NPP静态库
add_library(npp STATIC
    ${CPU_SOURCES}
    ${CUDA_SOURCES}
)

# 设置目标属性
set_target_properties(npp PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 链接CUDA库
target_link_libraries(npp 
    PUBLIC CUDA::cudart
)

# 设置编译选项
target_compile_features(npp PUBLIC cxx_std_17)
target_compile_options(npp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr>
)

# 导出包含目录
target_include_directories(npp PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../api
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)