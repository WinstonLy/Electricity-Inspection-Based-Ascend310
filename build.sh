#!/bin/bash
sudo rm -rf build
mkdir build
cd build
cmake ../src -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ -DCMAKE_SKIP_RPATH=TRUE
make 
cd ..
