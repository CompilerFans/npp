#!/bin/bash

# Compile the V1 vs V2 comparison test

g++ -o compare_v1_v2_npp compare_v1_v2_npp.cpp \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 \
    -lnppig -lnppicc -lnppidei -lcudart \
    -std=c++11 \
    -O2

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "Run with: ./compare_v1_v2_npp"
else
    echo "Compilation failed!"
    exit 1
fi
