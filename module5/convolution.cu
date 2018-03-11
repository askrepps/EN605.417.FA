/////////////////////////
// convolution.cu      // 
// Andrew Krepps       //
// Module 5 Assignment //
// 3/5/2018            //
/////////////////////////

#include <chrono>
#include <stdio.h>
#include <stdlib.h>

#define MAX_WEIGHTS 4096

///////////////////////////////////////////////////////////////////////////////
/// \brief perform convolution operation for a single output element
/// 
/// \param [in] inVec the input data vector
/// \param [in] inWeights the input weight vector
/// \param [out] outVec the output data vector
/// \param [in] numElements the number of input and output vector elements
/// \param [in] numWeights the number of weights
///////////////////////////////////////////////////////////////////////////////
__device__ void performConvolution(
	const float* inVec,
	const float* inWeights,
	float* outVec,
	const unsigned int numElements,
	const unsigned int numWeights)
{
	const unsigned int dataIdx = blockIdx.x*blockDim.x + threadIdx.x;
	if (dataIdx < numElements) {
		outVec[dataIdx] = 0.0f;
		for (unsigned int wIdx = 0; wIdx < numWeights; ++wIdx) {
			if (dataIdx + wIdx < numElements) {			
				outVec[dataIdx] += inVec[dataIdx+wIdx]*inWeights[wIdx];
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
/// \brief perform convolution using global memory
/// 
/// \param [in] inVec the input data vector
/// \param [in] inWeights the input weight vector
/// \param [out] outVec the output data vector
/// \param [in] numElements the number of input and output vector elements
/// \param [in] numWeights the number of weights
///////////////////////////////////////////////////////////////////////////////
__global__ void convolutionGlobalMem(
	const float* inVec,
	const float* inWeights,
	float* outVec,
	const unsigned int numElements,
	const unsigned int numWeights)
{
	// we're just using global memory, so directly perform convolution
	performConvolution(inVec, inWeights, outVec, numElements, numWeights);
}

///////////////////////////////////////////////////////////////////////////////
// \brief constant memory for storing convolution weights
///////////////////////////////////////////////////////////////////////////////
__constant__ float weightsConstantMem[MAX_WEIGHTS];

///////////////////////////////////////////////////////////////////////////////
/// \brief perform convolution using constant memory
/// 
/// This assumes that the weights have already been copied to constant memory
/// using the symbol weightsConstantMem.
/// 
/// \param [in] inVec the input data vector
/// \param [out] outVec the output data vector
/// \param [in] numElements the number of input and output vector elements
/// \param [in] numWeights the number of weights
///////////////////////////////////////////////////////////////////////////////
__global__ void convolutionConstantMem(
	const float* inVec,
	float* outVec,
	const unsigned int numElements,
	const unsigned int numWeights)
{
	// perform the convolution using weight array bound to constant memory
	performConvolution(inVec, weightsConstantMem, outVec, numElements, numWeights);
}

///////////////////////////////////////////////////////////////////////////////
/// \brief perform convolution using shared memory
/// 
/// \param [in] inVec the input data vector
/// \param [in] inWeights the input weight vector
/// \param [out] outVec the output data vector
/// \param [in] numElements the number of input and output vector elements
/// \param [in] numWeights the number of weights
///////////////////////////////////////////////////////////////////////////////
__global__ void convolutionSharedMem(
	const float* inVec,
	const float* inWeights,
	float* outVec,
	const unsigned int numElements,
	const unsigned int numWeights)
{
	// load weights into shared memory for each block
	extern __shared__ float weightsSharedMem[];
	
	const unsigned int localIdx = threadIdx.x;
	if (localIdx < numWeights) {
		weightsSharedMem[localIdx] = inWeights[localIdx];
	}
	__syncthreads();
	
	// after all data is loaded, perform convolution using weights in shared memory	
	performConvolution(inVec, weightsSharedMem, outVec, numElements, numWeights);
}

///////////////////////////////////////////////////////////////////////////////
/// \brief launch a kernel to perform convolution
/// 
/// \param [in] kernel the kernel index (i.e., memory type) to use
/// \param [in] blockSize the number of threads per block to use
/// \param [in] inVec the input data vector
/// \param [in] inWeights the input weight vector
/// \param [out] outVec the output data vector
/// \param [in] numElements the number of input and output vector elements
/// \param [in] numWeights the number of weights
/// 
/// \returns the kernel execution time (in ms)
///////////////////////////////////////////////////////////////////////////////
float launchKernel(
	const unsigned int kernel,
	const unsigned int blockSize,
	const float* inVec,
	const float* inWeights,
	float* outVec,
	const unsigned int numElements,
	const unsigned int numWeights)
{
	const unsigned int numBlocks = numElements/blockSize;
	
	// start clock and launch kernel
	auto start = std::chrono::high_resolution_clock::now();
	switch (kernel) {
		case 0:
			convolutionGlobalMem<<<numBlocks, blockSize>>>(inVec, inWeights, outVec, numElements, numWeights);
			break;
		case 1:
			convolutionConstantMem<<<numBlocks, blockSize>>>(inVec, outVec, numElements, numWeights);
			break;
		case 2:
			convolutionSharedMem<<<numBlocks, blockSize, numWeights*sizeof(float)>>>(inVec, inWeights, outVec, numElements, numWeights);
			break;
		default:
			printf("Invalid kernel index: %d\n", kernel);
	}
	
	// calculate execution time in ms
	auto stop = std::chrono::high_resolution_clock::now();	
	std::chrono::duration<float> duration(stop - start);
	
	return duration.count()*1000.0f;
}

///////////////////////////////////////////////////////////////////////////////
/// \brief run a kernel to perform convolution and print timing results
/// 
/// \param [in] kernel the kernel index (i.e., memory type) to use
/// \param [in] inVec the input data vector
/// \param [in] inWeights the input weight vector
/// \param [out] outVec the output data vector
/// \param [in] numElements the number of input and output vector elements
/// \param [in] numWeights the number of weights
///////////////////////////////////////////////////////////////////////////////
void runTimingTest(
	const unsigned int kernel,
	const unsigned int blockSize,
	const float* inVec,
	const float* inWeights,
	float* outVec,
	const unsigned int numElements,
	const unsigned int numWeights)
{
	switch (kernel) {
		case 0:
			printf("Running global memory kernel\n");
			break;
		case 1:
			printf("Running constant memory kernel\n");
			break;
		case 2:
			printf("Running shared memory kernel\n");
			break;
		default:
			printf("Invalid kernel index: %d\n", kernel);
	}

	float ms = launchKernel(kernel, blockSize, inVec, inWeights, outVec, numElements, numWeights);
	printf("Kernel took %.6f ms to run\n", ms);
}

int main(int argc, char** argv)
{
	// configure run
	unsigned int numElements = 1024;
	unsigned int blockSize = 128;
	unsigned int numWeights = 8;
	
	if (argc > 1) {
		numElements = atoi(argv[1]);
	}
	if (argc > 2) {
		blockSize = atoi(argv[2]);
	}
	if (argc > 3) {
		numWeights = atoi(argv[3]);
	}
	
	if (numWeights > MAX_WEIGHTS) {
		numWeights = MAX_WEIGHTS;
		printf("Warning: numWeights exceeds maximum limit. Setting to %d.\n", numWeights);
	}
	
	// allocate memory
	const unsigned int dataBytes = numElements*sizeof(float);
	const unsigned int weightBytes = numWeights*sizeof(float);

	// initialize input data
	float* inVec = (float*)malloc(dataBytes);
	float* outVec = (float*)malloc(dataBytes);
	for (unsigned int i = 0; i < numElements; ++i) {
		inVec[i] = 1.0f*i;
	}

	// initialize weights
	float* inWeights = (float*)malloc(weightBytes);
	for (unsigned int i = 0; i < numWeights; ++i) {
		inWeights[i] = (float)(i+1)/numWeights;
		//printf("w[%d] = %.3f\n", i, inWeights[i]);
	}
	
	// allocate device memory
	float* d_inVec;
	float* d_outVec;
	float* d_inWeights;
	
	cudaMalloc((void**)&d_inVec, dataBytes);
	cudaMalloc((void**)&d_outVec, dataBytes);
	cudaMalloc((void**)&d_inWeights, weightBytes);

	// initialize weights in constant memory for later
	cudaMemcpyToSymbol(weightsConstantMem, inWeights, weightBytes);
	
	// copy data from host to device (kernel does not modify input, so we only have to do this once)
	cudaMemcpy(d_inVec, inVec, dataBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_inWeights, inWeights, weightBytes, cudaMemcpyHostToDevice);
	
	// dummy executions to avoid startup performance hit
	for (unsigned int kernel = 0; kernel < 3; ++kernel) {
		launchKernel(kernel, blockSize, d_inVec, d_inWeights, d_outVec, numElements, numWeights);
	}
	
	// run timing comparisons
	for (unsigned int kernel = 0; kernel < 3; ++kernel) {
		runTimingTest(kernel, blockSize, d_inVec, d_inWeights, d_outVec, numElements, numWeights);
	}

	// print output of last kernel
	cudaMemcpy(outVec, d_outVec, dataBytes, cudaMemcpyDeviceToHost);
	for (unsigned int i = 0; i < numElements; ++i) {
		//printf("outVec[%d] = %f\n", i, outVec[i]);
	}

	// free memory
	cudaFree(d_inVec);
	cudaFree(d_outVec);
	cudaFree(d_inWeights);
	free(inVec);
	free(inWeights);
	free(outVec);
}
