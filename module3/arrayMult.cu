// arrayMult.cu
// Andrew Krepps
// Module 3 Assignment
// 2/19/2018

#include <stdio.h>

#define MAX_ARRAY_SIZE 65536

// element-wise multiplication of n values from
// in1 and in2, with the result stored in out
__global__
void arrayMult(const float* in1, const float* in2, float* out, const unsigned int n)
{
	unsigned int dataIdx = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (dataIdx < n) {
		out[dataIdx] = in1[dataIdx]*in2[dataIdx];
	}
}

int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = 64;
	int blockSize = 16;
	
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}
	if (totalThreads > MAX_ARRAY_SIZE) {
		printf("Warning: Total thread count is greater than MAX_ARRAY_SIZE\n");
	}

	// initialize data
	float in1[MAX_ARRAY_SIZE];
	float in2[MAX_ARRAY_SIZE];
	for (int i = 0; i < totalThreads; ++i) {
		in1[i] = (float)i;
		in2[i] = 0.25f*i;
	}
	
	// allocate device memory
	const int dataSize = totalThreads*sizeof(float);
	float* in1d;
	float* in2d;
	float* outd;
	cudaMalloc((void**)&in1d, dataSize);
	cudaMalloc((void**)&in2d, dataSize);
	cudaMalloc((void**)&outd, dataSize);
	
	// copy data to device
	cudaMemcpy(in1d, in1, dataSize, cudaMemcpyHostToDevice);
	cudaMemcpy(in2d, in2, dataSize, cudaMemcpyHostToDevice);
	
	// execute kernel
	arrayMult<<<numBlocks, blockSize>>>(in1d, in2d, outd, totalThreads);

	// copy results to host
	float out[MAX_ARRAY_SIZE];
	cudaMemcpy(out, outd, dataSize, cudaMemcpyDeviceToHost);

	// display results
	for (int i = 0; i < totalThreads; ++i) {
		printf("%f * %f = %f\n", in1[i], in2[i], out[i]);
	}

	// clean up device memory
	cudaFree(in1d);
	cudaFree(in2d);
	cudaFree(outd); 

	return EXIT_SUCCESS;
}
