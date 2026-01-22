# optimize relative options

message("GPU_TYPE: ${GPU_TYPE}")
if (GPU_TYPE STREQUAL "NVIDIA_GPU")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
endif()


if (GPU_TYPE STREQUAL "METAX_GPU")
  if(WITHOUT_CU_BRIDGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
        -use-dim-intrinsic \
        -mllvm -metaxgpu-new-streg-abi=true")
    message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
        -use-dim-intrinsic \
        -mllvm -metaxgpu-new-streg-abi=true")
    message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
  endif()
endif()