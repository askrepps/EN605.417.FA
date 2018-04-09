#!/bin/sh

runTest()
{
	numBlocks=$1
	blockSize=$2
	numStreams=$3
	
	build/streamedMult $numBlocks $blockSize $numStreams
}

runAllStreamConfigs()
{
	numBlocks=$1
	
	echo "Running with $numBlocks blocks and a block size of 256..."
	runTest $numBlocks 256 1
	runTest $numBlocks 256 2
	runTest $numBlocks 256 4
	runTest $numBlocks 256 8
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
nvcc -o build/streamedMult streamedMult.cu
if [ $? -ne 0 ]; then
        echo 'Compilation failed' >&2
        exit 1
fi
echo ''

runAllStreamConfigs 256
runAllStreamConfigs 512
runAllStreamConfigs 1024
runAllStreamConfigs 2048
runAllStreamConfigs 4096
runAllStreamConfigs 8192
runAllStreamConfigs 16384
runAllStreamConfigs 32768
runAllStreamConfigs 65536
runAllStreamConfigs 131072
