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

# 设置通用编译选项
function(npp_set_common_compile_options target)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr>
    )
    
    # 添加警告作为错误选项
    if(NPP_WARNINGS_AS_ERRORS)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-Werror>
            )
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/WX>
            )
        endif()
        
        # CUDA警告作为错误
        if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-Werror all-warnings>
            )
        endif()
    endif()
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
    npp_set_common_compile_options(${target})
    npp_link_cuda(${target})
endfunction()