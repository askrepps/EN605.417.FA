#!/bin/sh

runThrustTest()
{
	dataSize=$1
	blockSize=$2
	frequency=$3
	samplingRate=$4
	
	echo "Running original analyzer with $dataSize samples and a block size of $blockSize..."
	build/freqAnalyzer_old $dataSize $blockSize $frequency $samplingRate
	
	echo "Running Thrust analyzer with $dataSize samples and a block size of $blockSize..."
	build/freqAnalyzer_thrust $dataSize $blockSize $frequency $samplingRate
	
	echo ""
}

runNppTest()
{
	scaleFactor=$1
	interpMode=$2
	
	echo "Rescaling image to $scaleFactor times original size using interp mode $interpMode..."
	build/imageResize_npp $scaleFactor $scaleFactor $interpMode
}

runAllNppInterps()
{
	scaleFactor=$1
	runNppTest $scaleFactor 1
	runNppTest $scaleFactor 2
	runNppTest $scaleFactor 4
	runNppTest $scaleFactor 16
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
nvcc -std=c++11 -o build/freqAnalyzer_old freqAnalyzer_old.cu -lcufft
if [ $? -ne 0 ]; then
        echo 'Compilation failed' >&2
        exit 1
fi
nvcc -std=c++11 -o build/freqAnalyzer_thrust -Wno-deprecated-declarations freqAnalyzer_thrust.cu -lcufft
if [ $? -ne 0 ]; then
        echo 'Compilation failed' >&2
        exit 1
fi
nvcc -std=c++11 -o build/imageResize_npp -Icommon/UtilNPP -Icommon/FreeImage/include -Lcommon/FreeImage/lib/linux/x86_64 imageResize_npp.cu -lnppisu_static -lnppig_static -lnppc_static -lfreeimage -lculibos
if [ $? -ne 0 ]; then
        echo 'Compilation failed' >&2
        exit 1
fi
nvcc -std=c++11 -o build/shortestPath_nvgraph shortestPath_nvgraph.cu -lnvgraph
if [ $? -ne 0 ]; then
        echo 'Compilation failed' >&2
        exit 1
fi
echo ''

echo 'Running Thrust tests...'
runThrustTest 512 256 64 512
runThrustTest 512 512 132 512
runThrustTest 8192 256 16 512
runThrustTest 8192 512 24 512
runThrustTest 131072 256 8 512
runThrustTest 131072 512 64 512
runThrustTest 2097152 256 72 512
runThrustTest 2097152 512 128 512
runThrustTest 33554432 256 1 512
runThrustTest 33554432 512 4 512

echo 'Running NPP tests...'
runAllNppInterps 1
runAllNppInterps 2
runAllNppInterps 4
runAllNppInterps 8
runAllNppInterps 16

echo 'Running nvGRAPH test...'
build/shortestPath_nvgraph
