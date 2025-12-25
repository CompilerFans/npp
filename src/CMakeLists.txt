# NPP core library CMake configuration
#
# =============================================================================
# Library Structure for NPP Compatibility
# =============================================================================
# libmppc.so/a contains all implementations (corresponds to libnppc.so)
# Other libraries are stubs that link to libmppc:
#
# NVIDIA NPP Libraries -> MPP Libraries:
#   libnppc.so    -> libmppc.so     (Core - contains ALL implementations)
#   libnppial.so  -> libmppial.so   (stub -> libmppc)
#   libnppicc.so  -> libmppicc.so   (stub -> libmppc)
#   libnppidei.so -> libmppidei.so  (stub -> libmppc)
#   libnppif.so   -> libmppif.so    (stub -> libmppc)
#   libnppig.so   -> libmppig.so    (stub -> libmppc)
#   libnppim.so   -> libmppim.so    (stub -> libmppc)
#   libnppist.so  -> libmppist.so   (stub -> libmppc)
#   libnppisu.so  -> libmppisu.so   (stub -> libmppc)
#   libnppitc.so  -> libmppitc.so   (stub -> libmppc)
#   libnpps.so    -> libmpps.so     (stub -> libmppc)
# =============================================================================

# Find source files
npp_find_all_sources(CPU_SOURCES CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR})

# =============================================================================
# Main library: libmppc (contains all implementations)
# Target name "npp" is kept for internal CMake references (tests depend on it)
# Output name is "mppc" -> generates libmppc.so/libmppc.a
# =============================================================================
add_library(npp
    ${CPU_SOURCES}
    ${CUDA_SOURCES}
)

# Set NPP target configuration
npp_setup_target(npp)

# Add NPP specific include directories
npp_add_common_includes(npp)

# Set library output name to mppc (generates libmppc.so/libmppc.a)
set_target_properties(npp PROPERTIES
    OUTPUT_NAME "mppc"
    EXPORT_NAME "npp"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    -use-dim-intrinsic \
    -mllvm -metaxgpu-new-streg-abi=true")

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

# Install main library
npp_install_library(npp)

# =============================================================================
# Stub Libraries (link to libmppc via npp target)
# =============================================================================
# Define stub library names (these link to npp target which outputs libmppc)
set(MPP_STUB_LIBS
    mppial
    mppicc
    mppidei
    mppif
    mppig
    mppim
    mppist
    mppisu
    mppitc
    mpps
)

# Create stub source file
set(STUB_SOURCE_FILE "${CMAKE_CURRENT_BINARY_DIR}/mpp_stub.cpp")
file(WRITE ${STUB_SOURCE_FILE} "// MPP stub library - links to libmppc\n")

# Create each stub library (shared or static based on BUILD_SHARED_LIBS)
foreach(stub_name ${MPP_STUB_LIBS})
    add_library(${stub_name} ${STUB_SOURCE_FILE})

    # Link to main npp target (which outputs libmppc)
    target_link_libraries(${stub_name} PUBLIC npp)

    # Set output name and properties
    set_target_properties(${stub_name} PROPERTIES
        OUTPUT_NAME "${stub_name}"
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )

    # Add shared library specific properties
    if(BUILD_SHARED_LIBS)
        set_target_properties(${stub_name} PROPERTIES
            INSTALL_RPATH "$ORIGIN"
            INSTALL_RPATH_USE_LINK_PATH TRUE
        )
    endif()

    # Install stub library
    install(TARGETS ${stub_name}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT libs
    )
endforeach()

message(STATUS "MPP main library: libmppc (target: npp, contains all implementations)")
message(STATUS "MPP stub libraries: ${MPP_STUB_LIBS} (link to libmppc)")