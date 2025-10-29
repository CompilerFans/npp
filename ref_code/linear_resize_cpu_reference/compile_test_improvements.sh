#!/bin/bash

g++ -o test_v1_improvements test_v1_improvements.cpp \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 \
    -lnppig -lnppicc -lnppidei -lcudart \
    -std=c++11 \
    -O2

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "Run with: ./test_v1_improvements"
else
    echo "Compilation failed!"
    exit 1
fi
