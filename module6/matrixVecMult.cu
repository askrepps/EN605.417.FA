/////////////////////////
// matrixVecMult.cu    // 
// Andrew Krepps       //
// Module 6 Assignment //
// 3/12/2018           //
/////////////////////////

#include <chrono>
#include <stdio.h>
#include <stdlib.h>

#define MAX_SIZE 8192

///////////////////////////////////////////////////////////////////////////////
/// \brief perform matrix vector multiplication for a single output element
/// 
/// \param [in] inMat the input matrix
/// \param [in] inVec the input vector
/// \param [out] outVec the output vector
/// \param [in] m the number of matrix rows and the output vector length
/// \param [in] n the number of matrix columns and the input vector length
///////////////////////////////////////////////////////////////////////////////
__device__ void performMatVecMult(
	const float* inMat,
	const float* inVec,
	float* outVec,
	const unsigned int m,
	const unsigned int n)
{
	const unsigned int outIdx = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (outIdx < m) {
		// intermediate results are stored in registers
		// before being written back to the output
		float sum = 0.0f;
		unsigned int matRowStart = outIdx*n;
		for (unsigned int i = 0; i < n; i++) {
			unsigned int matIdx = matRowStart + i;
			sum += inMat[matIdx]*inVec[i];
		}
		outVec[outIdx] = sum;
	}
}

///////////////////////////////////////////////////////////////////////////////
/// \brief perform matrix vector multiplication using global memory
/// 
/// \param [in] inMat the input matrix
/// \param [in] inVec the input vector
/// \param [out] outVec the output vector
/// \param [in] m the number of matrix rows and the output vector length
/// \param [in] n the number of matrix columns and the input vector length
///////////////////////////////////////////////////////////////////////////////
__global__ void matVecMultGlobalMem(
	const float* inMat,
	const float* inVec,
	float* outVec,
	const unsigned int m,
	const unsigned int n)
{
	// we're just using global memory, so directly perform multiplication
	performMatVecMult(inMat, inVec, outVec, m, n);
}

///////////////////////////////////////////////////////////////////////////////
// \brief constant memory for storing input vector
///////////////////////////////////////////////////////////////////////////////
__constant__ float inVecConstantMem[MAX_SIZE];

///////////////////////////////////////////////////////////////////////////////
/// \brief perform matrix vector mutiplication using constant memory
/// 
/// This assumes that the input vector has already been copied to constant
/// memory using the symbol inVecConstantMem
/// 
/// \param [in] inMat the input matrix
/// \param [out] outVec the output vector
/// \param [in] m the number of matrix rows and the output vector length
/// \param [in] n the number of matrix columns and the input vector length
///////////////////////////////////////////////////////////////////////////////
__global__ void matVecMultConstantMem(
	const float* inMat,
	float* outVec,
	const unsigned int m,
	const unsigned int n)
{
	// perform the multiplication using input vector bound to constant memory
	performMatVecMult(inMat, inVecConstantMem, outVec, m, n);
}

///////////////////////////////////////////////////////////////////////////////
/// \brief perform matrix vector multiplication using shared memory
/// 
/// \param [in] inMat the input matrix
/// \param [in] inVec the input vector
/// \param [out] outVec the output vector
/// \param [in] m the number of matrix rows and the output vector length
/// \param [in] n the number of matrix columns and the input vector length
///////////////////////////////////////////////////////////////////////////////
__global__ void matVecMultSharedMem(
	const float* inMat,
	const float* inVec,
	float* outVec,
	const unsigned int m,
	const unsigned int n)
{
	// load input vector into shared memory for each block
	extern __shared__ float inVecSharedMem[];
	
	const unsigned int localIdx = threadIdx.x;
	
	// input vector could be larger than the block size,
	// so we need to figure out which elements each thread
	// is responsible for copying over to shared memory
	const unsigned int elementsToCopy = n/blockDim.x + 1;
	const unsigned int startElement = localIdx*elementsToCopy;
	for (unsigned int i = 0; i < elementsToCopy; ++i) {
		unsigned int dataIdx = startElement + i;
		if (dataIdx < n) {
			inVecSharedMem[dataIdx] = inVec[dataIdx];
		}
	}
	
	__syncthreads();
	
	// after all data is loaded, perform multiplication using vector in shared memory	
	performMatVecMult(inMat, inVecSharedMem, outVec, m, n);
}

///////////////////////////////////////////////////////////////////////////////
/// \brief initialize input data on the host
/// 
/// \param [out] mat the input matrix
/// \param [out] vec the input vector
/// \param [in] m the number of matrix rows
/// \param [in] n the number of matrix columns and the input vector length
///////////////////////////////////////////////////////////////////////////////
void initializeInputData(
	float* mat,
	float* vec,
	const unsigned int m,
	const unsigned int n)
{
	for (unsigned int i = 0; i < m; ++i) {
		for (unsigned int j = 0; j < n; ++j) {
			const unsigned int matIdx = i*n + j;
			mat[matIdx] = matIdx*0.01f;
		}
		vec[i] = i*0.1f;
	}
}

///////////////////////////////////////////////////////////////////////////////
/// \brief launch a kernel to perform matrix vector multiplication
/// 
/// \param [in] kernel the kernel index (i.e., memory type) to use
/// \param [in] blockSize the number of threads per block to use
/// \param [in] inMat the input matrix (on the device)
/// \param [in] inVec the input vector (on the device)
/// \param [out] outVec the output vector (on the device)
/// \param [in] m the number of matrix rows and the output vector length
/// \param [in] n the number of matrix columns and the input vector length
/// 
/// \returns the kernel execution time (in ms)
///////////////////////////////////////////////////////////////////////////////
float launchKernel(
	const unsigned int kernel,
	const unsigned int blockSize,
	const float* inMat,
	const float* inVec,
	float* outVec,
	const unsigned int m,
	const unsigned int n)
{
	const unsigned int numBlocks = m/blockSize;
	
	// start clock and launch kernel
	auto start = std::chrono::high_resolution_clock::now();
	switch (kernel) {
		case 0:
			matVecMultGlobalMem<<<numBlocks, blockSize>>>(inMat, inVec, outVec, m, n);
			break;
		case 1:
			matVecMultConstantMem<<<numBlocks, blockSize>>>(inMat, outVec, m, n);
			break;
		case 2:
			matVecMultSharedMem<<<numBlocks, blockSize, n*sizeof(float)>>>(inMat, inVec, outVec, m, n);
			break;
		default:
			printf("Invalid kernel index: %d\n", kernel);
	}
	
	// wait for GPU kernel to finish
	cudaThreadSynchronize();
	
	// calculate execution time in ms
	auto stop = std::chrono::high_resolution_clock::now();	
	std::chrono::duration<float> duration(stop - start);
	
	return duration.count()*1000.0f;
}

///////////////////////////////////////////////////////////////////////////////
/// \brief allocate device memory and run a kernel to perform matrix vector
/// multiplication
/// 
/// \param [in] kernel the kernel index (i.e., memory type) to use
/// \param [in] kernel the kernel index (i.e., memory type) to use
/// \param [in] blockSize the number of threads per block to use
/// \param [in] inMat the input matrix (on the host)
/// \param [in] inVec the input vector (on the host)
/// \param [out] outVec the output vector (on the host)
/// \param [in] m the number of matrix rows and the output vector length
/// \param [in] n the number of matrix columns and the input vector length
/// 
/// \returns the kernel execution time (in ms) not including data transfer
///////////////////////////////////////////////////////////////////////////////
float runTimingTest(
	const unsigned int kernel,
	const unsigned int blockSize,
	const float* inMat,
	const float* inVec,
	float* outVec,
	const unsigned int m,
	const unsigned int n)
{
	// allocate device memory
	float* d_inMat;
	float* d_inVec;
	float* d_outVec;
	
	const unsigned int matrixElements = m*n;
	const unsigned int matrixBytes = matrixElements*sizeof(float);
	const unsigned int vectorBytes = n*sizeof(float);
	
	cudaMalloc((void**)&d_inMat, matrixBytes);
	cudaMalloc((void**)&d_inVec, vectorBytes);
	cudaMalloc((void**)&d_outVec, vectorBytes);
	
	// copy input data to device
	cudaMemcpy(d_inMat, inMat, matrixBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_inVec, inVec, vectorBytes, cudaMemcpyHostToDevice);

	float ms = launchKernel(kernel, blockSize, d_inMat, d_inVec, d_outVec, m, n);

	// copy output data to host
	cudaMemcpy(outVec, d_outVec, vectorBytes, cudaMemcpyDeviceToHost);

	// free device memory
	cudaFree(d_inMat);
	cudaFree(d_inVec);
	cudaFree(d_outVec);
	
	return ms;
}

int main(int argc, char** argv)
{
	// configure run
	unsigned int dataSize = 512;
	unsigned int blockSize = 256;
	
	if (argc > 1) {
		dataSize = atoi(argv[1]);
	}
	if (argc > 2) {
		blockSize = atoi(argv[2]);
	}
	
	if (dataSize > MAX_SIZE) {
		dataSize = MAX_SIZE;
		printf("Warning: data size exceeds maximum limit. Setting to %d.\n", dataSize);
	}
	
	// allocate and initialize host memory
	const unsigned int matrixBytes = dataSize*dataSize*sizeof(float);
	const unsigned int vectorBytes = dataSize*sizeof(float);
	float* inMat = (float*)malloc(matrixBytes);
	float* inVec = (float*)malloc(vectorBytes);
	float* outVec = (float*)malloc(vectorBytes);
	initializeInputData(inMat, inVec, dataSize, dataSize);

	// initialize input vector in constant memory for later
	cudaMemcpyToSymbol(inVecConstantMem, inVec, vectorBytes);
	
	// dummy executions to avoid startup performance hit
	for (unsigned int kernel = 0; kernel < 3; ++kernel) {
		runTimingTest(kernel, blockSize, inMat, inVec, outVec, dataSize, dataSize);
	}
	
	// run timing comparisons
	for (unsigned int kernel = 0; kernel < 3; ++kernel) {
		switch (kernel) {
			case 0:
				printf("Running global memory kernel:   ");
				break;
			case 1:
				printf("Running constant memory kernel: ");
				break;
			case 2:
				printf("Running shared memory kernel    ");
				break;
			default:
				printf("Invalid kernel index: %d\n", kernel);
		}
		
		float ms = runTimingTest(kernel, blockSize, inMat, inVec, outVec, dataSize, dataSize);
		printf("Kernel took %.6f ms to run\n", ms);
			
		// print output of kernel
		for (unsigned int i = 0; i < dataSize; ++i) {
			//printf("outVec[%d] = %f\n", i, outVec[i]);
		}
	}

	// free host memory	
	free(inMat);
	free(inVec);
	free(outVec);
}
