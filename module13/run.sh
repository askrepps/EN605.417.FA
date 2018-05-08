#!/bin/sh

runTest()
{
    build/pipeline
    echo ""
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
echo ''

echo 'Running tests...'
runTest
