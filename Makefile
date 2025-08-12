# Simple Makefile for OpenNPP implementation

# Compiler and flags
CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++17 -Wall -Wextra -I./src/include -I./api
NVCCFLAGS = -std=c++17 -lineinfo -arch=sm_60 -I./src/include -I./api

# Directories
SRC_DIR = src
TEST_DIR = test
BUILD_DIR = build

# Files
CPU_SOURCES = $(SRC_DIR)/modules/npp_image_arithmetic_cpu.cpp
CUDA_SOURCES = $(SRC_DIR)/modules/npp_image_arithmetic_cuda.cu
TEST_SOURCES = $(TEST_DIR)/test_nppi_addc_8u_c1rsfs.cpp $(TEST_DIR)/test_nppi_addc_8u_c1rsfs_validation.cpp

# Targets
.PHONY: all clean test build validation

all: build test

# Build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build CPU library
build-cpu: $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $(CPU_SOURCES) -o $(BUILD_DIR)/npp_cpu.o
	ar rcs $(BUILD_DIR)/libopen_npp_cpu.a $(BUILD_DIR)/npp_cpu.o

# Build CUDA library
build-cuda: $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $(CUDA_SOURCES) -o $(BUILD_DIR)/npp_cuda.o
	ar rcs $(BUILD_DIR)/libopen_npp_cuda.a $(BUILD_DIR)/npp_cuda.o

# Build test executables
test: build-cpu build-cuda
	$(CXX) $(CXXFLAGS) -o $(BUILD_DIR)/test_nppi_addc $(TEST_DIR)/test_nppi_addc_8u_c1rsfs.cpp $(BUILD_DIR)/libopen_npp_cpu.a -lcudart
	$(NVCC) $(NVCCFLAGS) -o $(BUILD_DIR)/test_nppi_addc_validation $(TEST_DIR)/test_nppi_addc_8u_c1rsfs_validation.cpp $(BUILD_DIR)/npp_cpu.o $(BUILD_DIR)/npp_cuda.o -lnppial -lcudart

# CMake build
build:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. && make -j$(shell nproc)

# Run tests
test-run:
	$(BUILD_DIR)/test_nppi_addc
	$(BUILD_DIR)/test_nppi_addc_validation

# Run validation
validation: test
	$(BUILD_DIR)/test_nppi_addc_validation

# Clean
clean:
	rm -rf $(BUILD_DIR)

# Help
help:
	@echo "Available targets:"
	@echo "  all        - Build everything and run tests"
	@echo "  build      - Build using CMake"
	@echo "  test       - Build test executables"
	@echo "  test-run   - Run tests"
	@echo "  validation - Run validation against NVIDIA NPP"
	@echo "  clean      - Clean build directory"
	@echo "  help       - Show this help"