/////////////////////////
// streamedMult.cu     //
// Andrew Krepps       //
// Module 7 Assignment //
// 3/26/2018           //
/////////////////////////

#include <stdio.h>
#include <stdlib.h>

///////////////////////////////////////////////////////////////////////////////
/// \brief perform element-wise array multiplication
/// 
/// \param [in] in1 the first input array
/// \param [in] in2 the second input array
/// \param [out] out the output array
/// \param [in] n the number of array elements
///////////////////////////////////////////////////////////////////////////////
__global__
void arrayMult(const float* in1, const float* in2, float* out, const unsigned int n)
{
	unsigned int dataIdx = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (dataIdx < n) {
		out[dataIdx] = in1[dataIdx]*in2[dataIdx];
	}
}

///////////////////////////////////////////////////////////////////////////////
/// \brief initialize host array data
/// 
/// \param [out] a the data array
/// \param [in] n the number of array elements
///////////////////////////////////////////////////////////////////////////////
void initHostArray(float* a, const unsigned int n)
{
	for (unsigned int i = 0; i < n; ++i) {
		a[i] = (float)i;
	}
}

///////////////////////////////////////////////////////////////////////////////
/// \brief verify output data
/// 
/// This assumes both input arrays were initialized using initHostArray.
/// 
/// \param [in] a the output array
/// \param [in] n the number of array elements
///////////////////////////////////////////////////////////////////////////////
void verifyResult(float* a, const unsigned int n)
{
	for (unsigned int i = 0; i < n; ++i) {
		float expected = (float)i * (float)i;
		if (a[i] != expected) {
			printf("Error! a[%d] != %f\n", expected);
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
/// \brief perform element-wise array multiplication while evenly splitting
/// work among concurrent streams
/// 
/// \param [in] numBlocks the number of thread blocks (total)
/// \param [in] blockSize the number of threads per block
/// \param [in] numStreams the number of streams
/// 
/// \returns the total GPU memory copy and execution time (in ms)
///////////////////////////////////////////////////////////////////////////////
float runStreamedArrayMult(const unsigned int numBlocks, const unsigned int blockSize, const unsigned int numStreams)
{
	// calculate data size
	const unsigned int n = numBlocks*blockSize;
	const unsigned int numBytes = n*sizeof(float);
	
	// evenly split blocks among streams
	const unsigned int blocksPerStream = numBlocks/numStreams;
	const unsigned int elementsPerStream = n/numStreams;
	const unsigned int bytesPerStream = elementsPerStream*sizeof(float);
	
	// allocate and initialize pinned host memory
	float* in1;
	float* in2;
	float* out;
	cudaMallocHost((void**)&in1, numBytes);
	cudaMallocHost((void**)&in2, numBytes);
	cudaMallocHost((void**)&out, numBytes);
	initHostArray(in1, n);
	initHostArray(in2, n);
	
	// allocate device memory
	float* d_in1;
	float* d_in2;
	float* d_out;
	cudaMalloc((void**)&d_in1, numBytes);
	cudaMalloc((void**)&d_in2, numBytes);
	cudaMalloc((void**)&d_out, numBytes);
	
	// create streams
	cudaStream_t* streams = (cudaStream_t*) malloc(numStreams*sizeof(cudaStream_t));
	for (unsigned int i = 0; i < numStreams; ++i) {
		cudaStreamCreate(&streams[i]);
	}
	
	// create timing events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	// start timer
	cudaEventRecord(start, 0);
	
	// execute kernels (data is split evenly among streams)
	for (unsigned int i = 0; i < numStreams; ++i) {
		const unsigned int startIdx = i*elementsPerStream;
		cudaMemcpyAsync(d_in1 + startIdx, in1 + startIdx, bytesPerStream, cudaMemcpyHostToDevice, streams[i]);
		cudaMemcpyAsync(d_in2 + startIdx, in2 + startIdx, bytesPerStream, cudaMemcpyHostToDevice, streams[i]);
		arrayMult<<<blocksPerStream, blockSize, 0, streams[i]>>>(d_in1 + startIdx, d_in2 + startIdx, d_out + startIdx, elementsPerStream);
		cudaMemcpyAsync(out + startIdx, d_out + startIdx, bytesPerStream, cudaMemcpyDeviceToHost, streams[i]);
	}
	
	// stop timer and wait for GPU to finish
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	// calculate execution time (in ms)
	float ms;
	cudaEventElapsedTime(&ms, start, stop);
	
	// verify output
	verifyResult(out, n);

	// free events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	// free streams
	for (unsigned int i = 0; i < numStreams; ++i) {
		cudaStreamDestroy(streams[i]);
	}
	free(streams);
	
	// free device memory
	cudaFree(d_in1);
	cudaFree(d_in2);
	cudaFree(d_out);
	
	// free pinned host memory
	cudaFreeHost(in1);
	cudaFreeHost(in2);
	cudaFreeHost(out);
	
	// return execution time (in ms)
	return ms;
}

int main(int argc, char** argv)
{
	// configure run
	unsigned int numBlocks = 512;
	unsigned int blockSize = 256;
	unsigned int numStreams = 1;
	
	if (argc > 1) {
		numBlocks = atoi(argv[1]);
	}
	if (argc > 2) {
		blockSize = atoi(argv[2]);
	}
	if (argc > 3) {
		numStreams = atoi(argv[3]);
	}
	
	// dummy execution to avoid startup performance hits
	runStreamedArrayMult(numBlocks, blockSize, numStreams);
	
	// run experiment and display timing results
	float ms = runStreamedArrayMult(numBlocks, blockSize, numStreams);
	printf("GPU execution time (%d streams): %.3f ms\n", numStreams, ms);
	
	return EXIT_SUCCESS;
}
