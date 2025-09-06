# NPP核心库CMake配置

# 查找源文件
npp_find_all_sources(CPU_SOURCES CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR})

# 创建NPP静态库
add_library(npp STATIC
    ${CPU_SOURCES}
    ${CUDA_SOURCES}
)

# 设置NPP目标配置
npp_setup_target(npp)

# 添加NPP特有包含目录
npp_add_common_includes(npp)