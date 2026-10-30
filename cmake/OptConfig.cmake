# optimize relative options

message("GPU_TYPE: ${GPU_TYPE}")

if (GPU_TYPE STREQUAL "METAX_GPU" AND NOT USE_NVIDIA_NPP)
  set(MXCC_OPT_FLAGS "-use-dim-intrinsic \
                      -mllvm -metaxgpu-new-streg-abi=true"
                    )

  if(WITHOUT_CU_BRIDGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
        ${MXCC_OPT_FLAGS}")
    message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
        ${MXCC_OPT_FLAGS}")
    message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
  endif()
endif()
