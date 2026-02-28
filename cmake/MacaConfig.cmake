# MACA configuration helpers (manual via MACA_PATH)

# Configure MACA SDK environment.
function(npp_configure_maca)
    if(DEFINED ENV{MACA_PATH})
        set(MACA_PATH "$ENV{MACA_PATH}" CACHE PATH "MACA SDK root path")
    endif()

    if(NOT MACA_PATH)
        message(FATAL_ERROR "MACA_PATH is not set; please export MACA_PATH to your MACA SDK root")
    endif()

    set(MACA_INCLUDE_DIR "${MACA_PATH}/include" CACHE PATH "MACA include dir")
    set(MACA_LIB_DIR "${MACA_PATH}/lib" CACHE PATH "MACA lib dir")
    set(MACA_COMPILER "${MACA_PATH}/mxgpu_llvm/bin/mxcc" CACHE FILEPATH "MACA compiler")

    message(STATUS "MACA_PATH: ${MACA_PATH}")
    message(STATUS "MACA Include Dir: ${MACA_INCLUDE_DIR}")
    message(STATUS "MACA Library Dir: ${MACA_LIB_DIR}")
    message(STATUS "MACA Compiler: ${MACA_COMPILER}")

    set(NPP_SDK_INCLUDE_DIR ${MACA_INCLUDE_DIR} CACHE PATH "MACA include dir")
    set(NPP_SDK_LIB_DIR ${MACA_LIB_DIR} CACHE PATH "MACA library dirs")
    set(NPP_COMPILER ${MACA_COMPILER} CACHE FILEPATH "MACA compiler")
    set(NPP_SDK_LINK_LIBS "" CACHE STRING "MACA link libraries")
endfunction()
