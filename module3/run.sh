#!/bin/sh

runTest()
{
	numThreads=$1
	blockSize=$2

	echo "Running with $numThreads total threads and a block size of $blockSize threads each..."
	build/arrayMult $numThreads $blockSize > results.txt
	diff "expected_$numThreads.txt" results.txt
	result=$?
	rm results.txt
	
	if [ $result -eq 0 ]; then
		echo 'Test passed!'
	else
		echo 'Test failed!' >&2
		exit 1
	fi
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
nvcc -o build/arrayMult arrayMult.cu
if [ $? -ne 0 ]; then
        echo 'Compilation failed' >&2
        exit 1
fi

runTest 64 64
runTest 64 32
runTest 64 16
runTest 128 64
runTest 128 32
runTest 128 16
runTest 256 64
runTest 256 32
runTest 256 16
