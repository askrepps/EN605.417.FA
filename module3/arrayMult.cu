// arrayMult.cu
// Andrew Krepps
// Module 3 Assignment
// 2/18/2018

#include <stdio.h>

#define ARRAY_SIZE 1024

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
	// initialize data
	float in1[ARRAY_SIZE];
	float in2[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; ++i) {
		in1[i] = (float)i;
		in2[i] = 2.0f*i;
	}
	
	// allocate device memory
	const int dataSize = ARRAY_SIZE*sizeof(float);	
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
	arrayMult<<<16, 64>>>(in1d, in2d, outd, ARRAY_SIZE);

	// copy results to host
	float out[ARRAY_SIZE];
	cudaMemcpy(out, outd, dataSize, cudaMemcpyDeviceToHost);

	// display results
	for (int i = 0; i < ARRAY_SIZE; ++i) {
		printf("%f * %f = %f\n", in1[i], in2[i], out[i]);
	}

	// clean up device memory
	cudaFree(in1d);
	cudaFree(in2d);
	cudaFree(outd); 

	return EXIT_SUCCESS;
}
