# NPP项目公共配置

include(GNUInstallDirs)

# 设置通用编译标准
function(npp_set_common_standards target)
    set_target_properties(${target} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
    )
endfunction()

# 添加NPP通用包含目录
function(npp_add_common_includes target)
    target_include_directories(${target} PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/API>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_include_directories(${target} PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )
endfunction()

# 链接CUDA库
function(npp_link_cuda target)
    target_link_libraries(${target} PUBLIC CUDA::cudart)
endfunction()

# 设置完整的NPP目标配置
function(npp_setup_target target)
    npp_set_common_standards(${target})
    npp_link_cuda(${target})
endfunction()
