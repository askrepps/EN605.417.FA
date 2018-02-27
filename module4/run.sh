#!/bin/sh

runTest()
{
	numThreads=$1
	blockSize=$2
	
	echo "Running with $numThreads total threads and a block size of $blockSize threads each..."
	build/arrayXor $numThreads $blockSize
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
nvcc -std=c++11 -o build/arrayXor arrayXor.cu
if [ $? -ne 0 ]; then
        echo 'Compilation failed' >&2
        exit 1
fi

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
runTest 16384 256
runTest 16384 512
runTest 32768 256
runTest 32768 512
runTest 65536 256
runTest 65536 512
runTest 131072 256
runTest 131072 512
runTest 262144 256
runTest 262144 512
runTest 524288 256
runTest 524288 512
runTest 1048576 256
runTest 1048576 512
