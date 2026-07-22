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
    find_package(CUDAToolkit REQUIRED)

    if(CUDAToolkit_FOUND)
        message(STATUS "Found CUDA Toolkit: ${CUDAToolkit_VERSION}")
        message(STATUS "CUDA Include Dirs: ${CUDAToolkit_INCLUDE_DIRS}")
        message(STATUS "CUDA Library Dir: ${CUDAToolkit_LIBRARY_DIR}")
    else()
        message(FATAL_ERROR "CUDA Toolkit not found")
    endif()
endfunction()


# 完整的CUDA环境设置
function(npp_configure_cuda)
    npp_check_cuda_environment()
    npp_setup_cuda_architectures()

    set(NPP_SDK_INCLUDE_DIR ${CUDAToolkit_INCLUDE_DIRS}CACHE PATH "CUDA include dirs")
    set(NPP_SDK_LIB_DIR ${CUDAToolkit_LIBRARY_DIR}CACHE PATH "CUDA library dir")
    set(NPP_COMPILER ${CUDAToolkit_COMPILER} CACHE FILEPATH "CUDA compiler")
    set(NPP_SDK_LINK_LIBS CUDA::cudart CACHE STRING "CUDA link libraries")
endfunction()

message("GPU_TYPE: ${GPU_TYPE}")
if (GPU_TYPE STREQUAL "NVIDIA_GPU")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
endif()
