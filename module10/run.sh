#!/bin/sh

runTest()
{
	arraySize=$1
	
	echo "Running test with array size of $arraySize..."
	build/HelloWorld $arraySize
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
g++ -std=c++11 -o build/HelloWorld HelloWorld.cpp -framework OpenCL
if [ $? -ne 0 ]; then
        echo 'Compilation failed' >&2
        exit 1
fi
echo ''

echo 'Running tests...'
runTest 256
runTest 1024
runTest 4096
runTest 16384
runTest 65536
runTest 262144
runTest 1048576
runTest 4194304
runTest 16777216
runTest 67108864