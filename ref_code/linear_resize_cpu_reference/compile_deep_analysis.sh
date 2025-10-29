#!/bin/bash

g++ -o deep_pixel_analysis deep_pixel_analysis.cpp \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 \
    -lnppig -lnppicc -lnppidei -lcudart \
    -std=c++11 \
    -O2

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "Run with: ./deep_pixel_analysis"
else
    echo "Compilation failed!"
    exit 1
fi
