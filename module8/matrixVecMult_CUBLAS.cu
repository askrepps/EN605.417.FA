////////////////////////////////
// matrixVecMult_CUBLAS.cu    // 
// Andrew Krepps              //
// Module 8 Assignment        //
// 4/2/2018                   //
////////////////////////////////

#include <chrono>
#include <stdio.h>
#include <stdlib.h>

#include <cublas.h>

#define CUBLAS_KERNEL 0
#define CUSTOM_KERNEL 1

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
			mat[matIdx] = matIdx*0.001f;
		}
		vec[i] = i*0.01f;
	}
}

///////////////////////////////////////////////////////////////////////////////
/// \brief allocate device memory and run a CUBLAS kernel that performs
/// matrix vector multiplication
/// 
/// \param [in] blockSize the number of threads per block to use
/// \param [in] inMat the input matrix (on the host)
/// \param [in] inVec the input vector (on the host)
/// \param [out] outVec the output vector (on the host)
/// \param [in] m the number of matrix rows and the output vector length
/// \param [in] n the number of matrix columns and the input vector length
///////////////////////////////////////////////////////////////////////////////
void runCublasKernel(
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
	
	cublasAlloc(m*n, sizeof(float), (void**)&d_inMat);
	cublasAlloc(n, sizeof(float), (void**)&d_inVec);
	cublasAlloc(m, sizeof(float), (void**)&d_outVec);
	
	// copy input data to device
	cublasSetMatrix(m, n, sizeof(float), inMat, m, d_inMat, m);
	cublasSetVector(n, sizeof(float), inVec, 1, d_inVec, 1);
	
	// run kernel (host data is in row-major order so set transpose for column-major CUBLAS)
	cublasSgemv('T', m, n, 1.0f, d_inMat, m, d_inVec, 1, 0.0f, d_outVec, 1);
	
	// copy output data to host
	cublasGetVector(m, sizeof(float), d_outVec, 1, outVec, 1);
	
	// free device memory
	cublasFree(d_inMat);
	cublasFree(d_inVec);
	cublasFree(d_outVec);
}

///////////////////////////////////////////////////////////////////////////////
/// \brief allocate device memory and run a custom kernel that performs
/// matrix vector multiplication
/// 
/// \param [in] blockSize the number of threads per block to use
/// \param [in] inMat the input matrix (on the host)
/// \param [in] inVec the input vector (on the host)
/// \param [out] outVec the output vector (on the host)
/// \param [in] m the number of matrix rows and the output vector length
/// \param [in] n the number of matrix columns and the input vector length
///////////////////////////////////////////////////////////////////////////////
void runCustomKernel(
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
	const unsigned int inVectorBytes = n*sizeof(float);
	const unsigned int outVectorBytes = m*sizeof(float);
	
	cudaMalloc((void**)&d_inMat, matrixBytes);
	cudaMalloc((void**)&d_inVec, inVectorBytes);
	cudaMalloc((void**)&d_outVec, outVectorBytes);
	
	// copy input data to device
	cudaMemcpy(d_inMat, inMat, matrixBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_inVec, inVec, inVectorBytes, cudaMemcpyHostToDevice);
	
	// launch kernel
	const unsigned int numBlocks = m/blockSize;
	matVecMultSharedMem<<<numBlocks, blockSize, n*sizeof(float)>>>(d_inMat, d_inVec, d_outVec, m, n);

	// copy output data to host
	cudaMemcpy(outVec, d_outVec, outVectorBytes, cudaMemcpyDeviceToHost);

	// free device memory
	cudaFree(d_inMat);
	cudaFree(d_inVec);
	cudaFree(d_outVec);
}

///////////////////////////////////////////////////////////////////////////////
/// \brief allocate device memory and run a kernel that performs matrix vector
/// multiplication
/// 
/// \param [in] kernel the index of the kernel to execute
/// \param [in] blockSize the number of threads per block to use
/// \param [in] inMat the input matrix (on the host)
/// \param [in] inVec the input vector (on the host)
/// \param [out] outVec the output vector (on the host)
/// \param [in] m the number of matrix rows and the output vector length
/// \param [in] n the number of matrix columns and the input vector length
/// 
/// \returns the total execution time (in ms) including data transfer
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
	// start clock
	auto start = std::chrono::high_resolution_clock::now();
	
	// run kernel
	switch (kernel) {
		case CUBLAS_KERNEL:
			runCublasKernel(blockSize, inMat, inVec, outVec, m, n);
			break;
		case CUSTOM_KERNEL:
			runCustomKernel(blockSize, inMat, inVec, outVec, m, n);
			break;
		default:
			printf("Error: unrecognized kernel index: %d\n", kernel);
	}
	
	// calculate execution time in ms
	auto stop = std::chrono::high_resolution_clock::now();	
	std::chrono::duration<float> duration(stop - start);
	
	return duration.count()*1000.0f;
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
	
	// allocate and initialize host memory
	const unsigned int matrixBytes = dataSize*dataSize*sizeof(float);
	const unsigned int vectorBytes = dataSize*sizeof(float);
	float* inMat = (float*)malloc(matrixBytes);
	float* inVec = (float*)malloc(vectorBytes);
	float* outVec = (float*)malloc(vectorBytes);
	initializeInputData(inMat, inVec, dataSize, dataSize);
	
	// initialze CUBLAS
	cublasInit();

	// run timing comparisons
	float cublasMs = runTimingTest(CUBLAS_KERNEL, blockSize, inMat, inVec, outVec, dataSize, dataSize);
	float customMs = runTimingTest(CUSTOM_KERNEL, blockSize, inMat, inVec, outVec, dataSize, dataSize);
	
	// show results
	printf("CUBLAS kernel time: %.3f ms\n", cublasMs);
	printf("Custom kernel time: %.3f ms\n", customMs);
	
	// shut down CUBLAS
	cublasShutdown();
	
	// free host memory
	free(inMat);
	free(inVec);
	free(outVec);
}
