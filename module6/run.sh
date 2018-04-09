#!/bin/sh

runTest()
{
	dataSize=$1
	blockSize=$2
	
	echo "Running with a $dataSize x $dataSize matrix and a block size of $blockSize..."
	build/matrixVecMult $dataSize $blockSize
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
nvcc -std=c++11 -o build/matrixVecMult matrixVecMult.cu
if [ $? -ne 0 ]; then
        echo 'Compilation failed' >&2
        exit 1
fi

runTest 256 256
runTest 512 256
runTest 512 512
runTest 1024 256
runTest 1024 512
runTest 2048 256
runTest 2048 512
runTest 4096 256
runTest 4096 512
runTest 8192 256
runTest 8192 512
