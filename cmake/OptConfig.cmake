# optimize relative options

message("GPU_TYPE: ${GPU_TYPE}")
if (GPU_TYPE STREQUAL "NVIDIA_GPU")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
endif()


if (GPU_TYPE STREQUAL "METAX_GPU")
  set(METAX_TARGET_ARCH_FLAGS "--offload-arch=xcore1000 --offload-arch=xcore1100 --offload-arch=xcore1500 --offload-arch=xcore1520 --offload-arch=xcore1600")
  set(MXCC_OPT_FLAGS "-use-dim-intrinsic \
                      -mllvm -metaxgpu-new-streg-abi=true"
                    )

  if(WITHOUT_CU_BRIDGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
        ${METAX_TARGET_ARCH_FLAGS} \
        ${MXCC_OPT_FLAGS}")
    message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
        ${METAX_TARGET_ARCH_FLAGS} \
        ${MXCC_OPT_FLAGS}")
    message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
  endif()
endif()