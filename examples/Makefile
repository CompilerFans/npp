CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc

# Include paths
INCLUDES = -I../API -I$(CUDA_PATH)/include

# Library paths
LDFLAGS = -L../build/lib -L$(CUDA_PATH)/lib64
LIBS = -lmpp -lcudart

# Compiler flags
CXXFLAGS = -std=c++11
NVCCFLAGS = $(CXXFLAGS) $(INCLUDES)

TARGET = nppiCopy_P3C3R_example

all: $(TARGET)

$(TARGET): nppiCopy_P3C3R_example.cpp
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

run: $(TARGET)
	LD_LIBRARY_PATH=../build/lib:$(LD_LIBRARY_PATH) ./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
