# CUDA配置功能

# 设置CUDA架构
function(npp_setup_cuda_architectures)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "80;89" CACHE STRING "CUDA architectures" FORCE)
        message(STATUS "Using default CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        message(STATUS "Using specified CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
endfunction()

# 检查CUDA环境
function(npp_check_cuda_environment)
    # CMake 3.17+ 支持 CUDAToolkit，3.10 需要用传统方法
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.17")
        find_package(CUDAToolkit REQUIRED)
        
        if(CUDAToolkit_FOUND)
            message(STATUS "Found CUDA Toolkit: ${CUDAToolkit_VERSION}")
            message(STATUS "CUDA Include Dirs: ${CUDAToolkit_INCLUDE_DIRS}")
            message(STATUS "CUDA Library Dir: ${CUDAToolkit_LIBRARY_DIR}")
            
            # 设置变量供其他地方使用
            set(CUDA_TOOLKIT_ROOT_DIR ${CUDAToolkit_LIBRARY_DIR}/.. PARENT_SCOPE)
        else()
            message(FATAL_ERROR "CUDA Toolkit not found")
        endif()
    else()
        # CMake 3.10-3.16: 使用传统的 CUDA 检测
        enable_language(CUDA)
        
        if(NOT DEFINED CMAKE_CUDA_COMPILER)
            message(FATAL_ERROR "CUDA compiler not found")
        endif()
        
        # 获取 CUDA 版本
        execute_process(
            COMMAND ${CMAKE_CUDA_COMPILER} --version
            OUTPUT_VARIABLE CUDA_VERSION_OUTPUT
            ERROR_QUIET
        )
        
        if(CUDA_VERSION_OUTPUT MATCHES "release ([0-9]+\\.[0-9]+)")
            set(CUDA_VERSION ${CMAKE_MATCH_1})
            message(STATUS "Found CUDA Toolkit: ${CUDA_VERSION}")
        else()
            message(STATUS "Found CUDA but could not determine version")
        endif()
        
        # 设置 CUDA 路径
        get_filename_component(CUDA_BIN_DIR ${CMAKE_CUDA_COMPILER} DIRECTORY)
        get_filename_component(CUDA_TOOLKIT_ROOT_DIR ${CUDA_BIN_DIR} DIRECTORY)
        
        message(STATUS "CUDA Include Dirs: ${CUDA_TOOLKIT_ROOT_DIR}/include")
        message(STATUS "CUDA Library Dir: ${CUDA_TOOLKIT_ROOT_DIR}/lib64")
        
        # 设置变量供其他地方使用
        set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR} PARENT_SCOPE)
        set(CUDA_LIBRARIES cuda cudart PARENT_SCOPE)
        set(CUDA_cudart_LIBRARY cudart PARENT_SCOPE)
    endif()
endfunction()


# 完整的CUDA环境设置
function(npp_configure_cuda)
    npp_check_cuda_environment()
    npp_setup_cuda_architectures()
endfunction()