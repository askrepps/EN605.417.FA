#!/bin/sh

runTests()
{
    echo 'Running tests...'
    echo ''
    echo '1 2 3 4 5' | build/pipeline
    echo ''
    echo '1 2 3 4 5 6 7 8' | build/pipeline 8
    echo ''
    build/pipeline 16 data/test16.txt
    echo ''
    build/pipeline 32 data/test32.txt
    echo ''
    build/pipeline 64 data/test64.txt
    echo ''
}

echo 'Creating build directory...'
if [ ! -d build ]; then
    mkdir build
    if [ $? -ne 0 ]; then
        echo 'Could not create build directory' >&2
        exit 1
    fi
fi

echo 'Compiling...'
CXXFLAGS='-std=c++11'
if [[ $OSTYPE == darwin* ]]; then
    LINKFLAGS='-framework OpenCL'
else
    CXXFLAGS="$CXXFLAGS -I/usr/local/cuda/include -L/usr/local/cuda/lib64"
    LINKFLAGS='-lOpenCL'
fi
g++ $CXXFLAGS -o build/pipeline pipeline.cpp $LINKFLAGS
if [ $? -ne 0 ]; then
        echo 'Compilation failed' >&2
        exit 1
fi

runTests
