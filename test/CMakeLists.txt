# NPP test CMake configuration

# Setup GoogleTest
npp_setup_gtest()

# Unit tests (pure functional tests, no external dependencies)
add_subdirectory(unit)

# Performance benchmarks (optional)
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
    message(STATUS "Performance benchmarks enabled")
else()
    message(STATUS "Performance benchmarks disabled (use -DBUILD_BENCHMARKS=ON to enable)")
endif()