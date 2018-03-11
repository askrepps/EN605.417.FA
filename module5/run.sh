#!/bin/sh

runTest()
{
	numThreads=$1
	blockSize=$2
	numWeights=$3
	
	echo "Running with $numThreads threads, a block size of $blockSize, and $numWeights weights..."
	build/convolution $numThreads $blockSize $numWeights
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
nvcc -std=c++11 -o build/convolution convolution.cu
if [ $? -ne 0 ]; then
        echo 'Compilation failed' >&2
        exit 1
fi

runTest 128 64 8
runTest 128 128 8
runTest 128 128 32
runTest 256 256 8
runTest 65536 256 8
runTest 262144 256 8
runTest 4194304 256 8
runTest 268435456 512 8
