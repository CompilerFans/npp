# NPP项目公共配置

include(GNUInstallDirs)

# 设置通用编译标准
function(npp_set_common_standards target)
    set_target_properties(${target} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    if(NOT USE_MACA_SDK)
        set_target_properties(${target} PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_SEPARABLE_COMPILATION ON
        )
    endif()
endfunction()

# 添加NPP通用包含目录
function(npp_add_common_includes target)
    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/API>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    if(USE_MACA_SDK)
        target_include_directories(${target} PRIVATE
            ${MACA_INCLUDE_DIR}
        )
    else()
        target_include_directories(${target} PRIVATE
            ${CUDAToolkit_INCLUDE_DIRS}
        )
    endif()
endfunction()

# 链接CUDA库
function(npp_link_cuda target)
    if(NOT USE_MACA_SDK)
        target_link_libraries(${target} PUBLIC CUDA::cudart)
    endif()
endfunction()

# link maca lib
function(npp_link_maca target)
    if(USE_MACA_SDK AND MACA_LIB_DIR)
        target_link_directories(${target} PUBLIC ${MACA_LIB_DIR})
    endif()
endfunction()

# 设置完整的NPP目标配置
function(npp_setup_target target)
    npp_set_common_standards(${target})
    if(USE_MACA_SDK)
        npp_link_maca(${target})
    else()
        npp_link_cuda(${target})
    endif()
endfunction()
