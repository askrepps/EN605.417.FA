#!/usr/bin/sh

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

echo 'Running...'
build/arrayMult > results.txt

echo 'Checking results...'
diff expected.txt results.txt
result=$?
rm results.txt

if [ $result -eq 0 ]; then
	echo 'Test passed!'
else
	echo 'Test failed!' >&2
	exit 1
fi
