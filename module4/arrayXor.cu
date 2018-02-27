/////////////////////////
// arrayXor.cu         // 
// Andrew Krepps       //
// Module 4 Assignment //
// 2/26/2018           //
/////////////////////////

#include <chrono>
#include <stdio.h>

///////////////////////////////////////////////////////////////////////////////
/// \brief calculate the bitwise exclusive OR of two arrays
/// 
/// \param [in] in1 the first input array
/// \param [in] in2 the second input array
/// \param [out] out the output array
/// \param [in] n the number of elements in each array
///////////////////////////////////////////////////////////////////////////////
__global__
void arrayXor(const unsigned int* in1, const unsigned int* in2, unsigned int* out, const unsigned int n)
{
	unsigned int dataIdx = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (dataIdx < n) {
		out[dataIdx] = in1[dataIdx]^in2[dataIdx];
	}
}

///////////////////////////////////////////////////////////////////////////////
/// \brief initialize input arrays in host memory
/// 
/// \param [out] hostIn1 the first input array
/// \param [out] hostIn2 the second input array
/// \param [in] n the number of elements in each array
///////////////////////////////////////////////////////////////////////////////
void initializeHostMemory(unsigned int* hostIn1, unsigned int* hostIn2, const unsigned int n)
{
	for (unsigned int i = 0; i < n; ++i) {
		hostIn1[i] = i;
		hostIn2[n-i] = i;
	}
}

///////////////////////////////////////////////////////////////////////////////
/// \brief run kernel execution with pageable host memory
/// 
/// \param [in] n the number of array elements
/// \param [in] numBlocks the number of blocks
/// \param [in] blockSize the number of threads per block
/// \param [in] the number of iterations to execute includes
/// (allocated memory will be reused between executions)
///////////////////////////////////////////////////////////////////////////////
void runWithPageableMemory(const unsigned int n, const unsigned int numBlocks, const unsigned int blockSize, const unsigned int iterations)
{
	unsigned int *in1, *in2, *out;
	unsigned int *d_in1, *d_in2, *d_out;
	
	// allocate pageable host memory
	unsigned int bytes = n*sizeof(unsigned int);
	in1 = (unsigned int*)malloc(bytes);
	in2 = (unsigned int*)malloc(bytes);
	out = (unsigned int*)malloc(bytes);
	
	// initialize host input data
	initializeHostMemory(in1, in2, n);
	
	// allocate device memory
	cudaMalloc((void**)&d_in1, bytes);
	cudaMalloc((void**)&d_in2, bytes);
	cudaMalloc((void**)&d_out, bytes);
	
	// run all iterations
	for (unsigned int i = 0; i < iterations; ++i) {
		// copy input data to device
		cudaMemcpy(d_in1, in1, bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(d_in2, in2, bytes, cudaMemcpyHostToDevice);
		
		// execute kernel
		arrayXor<<<numBlocks, blockSize>>>(d_in1, d_in2, d_out, n);
		
		// copy output data to host
		cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);
	}

	// free allocated memory
	cudaFree(d_in1);
	cudaFree(d_in2);
	cudaFree(d_out);
	free(in1);
	free(in2);
	free(out);
}

///////////////////////////////////////////////////////////////////////////////
/// \brief run kernel execution with pinned host memory
/// 
/// \param [in] n the number of array elements
/// \param [in] numBlocks the number of blocks
/// \param [in] blockSize the number of threads per block
/// \param [in] the number of iterations to execute includes
/// (allocated memory will be reused between executions)
///////////////////////////////////////////////////////////////////////////////
void runWithPinnedMemory(const unsigned int n, const unsigned int numBlocks, const unsigned int blockSize, const unsigned int iterations)
{
	unsigned int *in1, *in2, *out;
	unsigned int *d_in1, *d_in2, *d_out;
	
	// allocate pinned host memory
	unsigned int bytes = n*sizeof(unsigned int);
	cudaMallocHost((void**)&in1, bytes);
	cudaMallocHost((void**)&in2, bytes);
	cudaMallocHost((void**)&out, bytes);
	
	// initialize host input data
	initializeHostMemory(in1, in2, n);
	
	// allocate device memory
	cudaMalloc((void**)&d_in1, bytes);
	cudaMalloc((void**)&d_in2, bytes);
	cudaMalloc((void**)&d_out, bytes);
	
	// run all iterations
	for (unsigned int i = 0; i < iterations; ++i) {
		// copy input data to device
		cudaMemcpy(d_in1, in1, bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(d_in2, in2, bytes, cudaMemcpyHostToDevice);
		
		// execute kernel
		arrayXor<<<numBlocks, blockSize>>>(d_in1, d_in2, d_out, n);
		
		// copy output data to host
		cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);
	}

	// free allocated memory
	cudaFree(d_in1);
	cudaFree(d_in2);
	cudaFree(d_out);
	cudaFreeHost(in1);
	cudaFreeHost(in2);
	cudaFreeHost(out);
}

int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;
	
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
	
	// the first execution appears to take longer (possibly due to caching)
	// so run each memory type once first to avoid affecting the timing results
	runWithPageableMemory(totalThreads, numBlocks, blockSize, 1);
	runWithPinnedMemory(totalThreads, numBlocks, blockSize, 1);

	// run pageable host memory (1 iteration)	
	auto singlePageableStart = std::chrono::high_resolution_clock::now();
	runWithPageableMemory(totalThreads, numBlocks, blockSize, 1);
	auto singlePageableStop = std::chrono::high_resolution_clock::now();
	
	// run pinned host memory (1 iteration)
	auto singlePinnedStart = std::chrono::high_resolution_clock::now();
	runWithPinnedMemory(totalThreads, numBlocks, blockSize, 1);
	auto singlePinnedStop = std::chrono::high_resolution_clock::now();
	
	// run pageable host memory (100 iterations)
	auto multiPageableStart = std::chrono::high_resolution_clock::now();
	runWithPageableMemory(totalThreads, numBlocks, blockSize, 100);
	auto multiPageableStop = std::chrono::high_resolution_clock::now();
	
	// run pinned host memory (100 iterations)
	auto multiPinnedStart = std::chrono::high_resolution_clock::now();
	runWithPinnedMemory(totalThreads, numBlocks, blockSize, 100);
	auto multiPinnedStop = std::chrono::high_resolution_clock::now();

	// display timing results (in ms)
	std::chrono::duration<float> duration;

	duration = singlePageableStop - singlePageableStart;
	float singlePageableMs = duration.count()*1000.0f;
	printf("Pageable memory (1 iteration):    %.6f ms\n", singlePageableMs);
	
	duration = singlePinnedStop - singlePinnedStart;
	float singlePinnedMs = duration.count()*1000.0f;
	printf("Pinned memory (1 iteration):      %.6f ms\n", singlePinnedMs);
	
	duration = multiPageableStop - multiPageableStart;
	float multiPageableMs = duration.count()*1000.0f;
	printf("Pageable memory (100 iterations): %.6f ms\n", multiPageableMs);
	
	duration = multiPinnedStop - multiPinnedStart;
	float multiPinnedMs = duration.count()*1000.0f;
	printf("Pinned memory (100 iterations):   %.6f ms\n", multiPinnedMs);
	
	if (singlePageableMs < singlePinnedMs) {
		printf("1: Pageable wins");
	}
	else if (singlePageableMs > singlePinnedMs) {
		printf("1: Pinned wins");
	}
	else {
		printf("1: It's a tie");
	}
	if (multiPageableMs < multiPinnedMs) {
		printf(" | 100: Pageable wins\n\n");
	}
	else if (multiPageableMs > multiPinnedMs) {
		printf(" | 100: Pinned wins\n\n");
	}
	else {
		printf(" | 100: It's a tie\n\n");
	}
	
	return EXIT_SUCCESS;
}
