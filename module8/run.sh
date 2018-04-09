#!/bin/sh

runMultTest()
{
	dataSize=$1
	blockSize=$2
	
	echo "Running with a $dataSize x $dataSize matrix and a block size of $blockSize..."
	build/matrixVecMult_CUBLAS $dataSize $blockSize
	echo ""
}

runFftTest()
{
	dataSize=$1
	blockSize=$2
	frequency=$3
	samplingRate=$4
	
	echo "Running with $dataSize samples and a block size of $blockSize..."
	build/freqAnalyzer $dataSize $blockSize $frequency $samplingRate
	echo""
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
nvcc -std=c++11 -o build/matrixVecMult_CUBLAS matrixVecMult_CUBLAS.cu -lcublas
if [ $? -ne 0 ]; then
	echo 'Compilation failed' >&2
fi
nvcc -std=c++11 -o build/freqAnalyzer freqAnalyzer.cu -lcufft
if [ $? -ne 0 ]; then
        echo 'Compilation failed' >&2
        exit 1
fi
echo ''

echo 'Running CUBLAS tests...'
runMultTest 256 256
runMultTest 512 256
runMultTest 512 512
runMultTest 1024 256
runMultTest 1024 512
runMultTest 2048 256
runMultTest 2048 512
runMultTest 4096 256
runMultTest 4096 512
runMultTest 8192 256
runMultTest 8192 512

echo 'Running cuFFT tests...'
runFftTest 512 256 64 512
runFftTest 512 512 132 512
runFftTest 8192 256 16 512
runFftTest 8192 512 24 512
runFftTest 131072 256 8 512
runFftTest 131072 512 64 512
runFftTest 2097152 256 72 512
runFftTest 2097152 512 128 512
runFftTest 33554432 256 1 512
runFftTest 33554432 512 4 512
