# NPP core library CMake configuration

# Find source files
npp_find_all_sources(CPU_SOURCES CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR})

# Create MPP library (static or dynamic based on BUILD_SHARED_LIBS)
add_library(npp
    ${CPU_SOURCES}
    ${CUDA_SOURCES}
)

# Set NPP target configuration
npp_setup_target(npp)

# Add NPP specific include directories
npp_add_common_includes(npp)

# Set library export name and version info
set_target_properties(npp PROPERTIES
    OUTPUT_NAME "mpp"
    EXPORT_NAME "npp"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Add dynamic library specific configuration if shared
if(BUILD_SHARED_LIBS)
    # Windows DLL export symbols
    target_compile_definitions(npp PRIVATE MPP_EXPORTS)
    target_compile_definitions(npp INTERFACE MPP_IMPORTS)
    
    # Set runtime path for dynamic library
    if(NOT WIN32)
        set_target_properties(npp PROPERTIES
            INSTALL_RPATH "$ORIGIN"
            INSTALL_RPATH_USE_LINK_PATH TRUE
        )
    endif()
endif()

# Install library files
npp_install_library(npp)