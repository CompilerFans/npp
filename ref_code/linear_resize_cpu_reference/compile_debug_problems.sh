#!/bin/bash

g++ -o debug_problem_cases debug_problem_cases.cpp \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 \
    -lnppig -lnppicc -lnppidei -lcudart \
    -std=c++11 \
    -O2

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "Run with: ./debug_problem_cases"
else
    echo "Compilation failed!"
    exit 1
fi
